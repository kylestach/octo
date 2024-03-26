from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb
from itertools import product

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng
from functools import partial
from experiments.homer.sim.wrappers.mujoco import MujocoManipWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to Octo checkpoint directory.")
flags.DEFINE_string("name", None, "Checkpoint name.")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step")
flags.DEFINE_float("increment", 0.05, "camera shift increment")
flags.DEFINE_integer("num_steps", 2, "number of steps to increment")

CENTER_CAMERA_VIEW = [1.2, 0.0, 0.4]

NUM_ROLLOUTS = 10
TIMESTEPS = 50
CONTEXT_WINDOW_SIZE = 32
N_VIS_ROLLOUTS = 3
VIS_FPS = 10
VIDEO_SUBSAMPLE_RATE = 1

def main(_):
    # setup wandb for logging
    wandb.init(name=f"eval_{FLAGS.name}", project="octo")

    # load finetuned model
    logging.info("Loading model...")
    model = OctoModel.load_pretrained(FLAGS.checkpoint_path, FLAGS.checkpoint_step)

    camera_views = product(
        *[[CENTER_CAMERA_VIEW[i] + FLAGS.increment * j for j in range(-FLAGS.num_steps, FLAGS.num_steps + 1)] for i in range(3)]
    )
    camera_views = [np.round(cv, 2) for cv in camera_views]

    all_rollout_info = {}

    for camera_pos in camera_views:

        str_camera_pos = "_".join([str(x) for x in camera_pos])

        # make gym environment
        env = gym.make(
            "franka-shoe-pick-place",
            goals_path=f"gs://rail-tpus-homer-v4/mujoco_rlds/mujoco_manip/camera_views/camera_{str_camera_pos}/eval_goals.npy",
            context_path=f"gs://rail-tpus-homer-v4/mujoco_rlds/mujoco_manip/camera_views/camera_{str_camera_pos}/context_traj.npy",
            unnormalization_statistics=model.dataset_statistics["mujoco_manip"]["action"],
            base_env_kwargs={"camera_pos": camera_pos}
        )

        # add wrappers for history and action chunking
        env = HistoryWrapper(env, horizon=2, include_past_action=False)
        env = TemporalEnsembleWrapper(env, pred_horizon=4)

        policy_fn = supply_rng(partial(model.sample_actions))

        rollout_info = {
            "episode_returns": [],
            "episode_metrics": [],
        }

        def listdict2dictlist(LD):
            return {k: [dic[k] for dic in LD] for k in LD[0]}

        # running rollouts
        for rollout_idx in range(NUM_ROLLOUTS):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            task = model.create_tasks(goals=env.get_goal())
            task["context"] = env.get_context(CONTEXT_WINDOW_SIZE)

            images = [obs["image_primary"][-1]]
            episode_return = 0.0
            metrics = []
            while len(images) < TIMESTEPS:
                actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
                actions = actions[0]
                obs, reward, done, trunc, info = env.step(actions)
                if "observations" in info:
                    images.extend(
                        [o["image_primary"][-1] for o in info["observations"]]
                    )
                else:
                    images.append(obs["image_primary"][-1])
                episode_return += reward
                if "metrics" in info:
                    metrics.append(info["metrics"])
                if done or trunc:
                    break

            rollout_info["episode_returns"].append(episode_return)
            if metrics:
                # concatenate all chunks into one dict of lists, then average across episode
                metrics = listdict2dictlist(metrics)
                rollout_info["episode_metrics"].append(
                    jax.tree_map(lambda x: np.mean(x), metrics)
                )
            if hasattr(env, "get_episode_metrics"):
                if metrics:
                    rollout_info["episode_metrics"][-1].update(
                        env.get_episode_metrics()
                    )
                else:
                    rollout_info["episode_metrics"].append(
                        env.get_episode_metrics()
                    )
            if rollout_idx < N_VIS_ROLLOUTS:
                # save rollout video
                assert (
                    images[0].dtype == np.uint8
                ), f"Expect uint8, got {images[0].dtype}"
                assert (
                    images[0].shape[-1] == 3
                ), f"Expect [height, width, channels] format, got {images[0].shape}"
                images = [
                    np.concatenate([task["image_primary"][0], frame], axis=0)
                    for frame in images
                ]
                rollout_info[f"rollout_{rollout_idx}_vid"] = wandb.Video(
                    np.array(images).transpose(0, 3, 1, 2)[
                        :: VIDEO_SUBSAMPLE_RATE
                    ],
                    fps=VIS_FPS,
                )
        rollout_info["avg_return"] = np.mean(rollout_info["episode_returns"])
        rollout_info["episode_returns"] = wandb.Histogram(
            rollout_info["episode_returns"]
        )
        if rollout_info["episode_metrics"]:
            metrics = listdict2dictlist(rollout_info.pop("episode_metrics"))
            for metric in metrics:
                rollout_info[metric] = wandb.Histogram(metrics[metric])
                rollout_info[f"avg_{metric}"] = np.mean(metrics[metric])
        else:
            rollout_info.pop("episode_metrics")

        all_rollout_info[f"camera_pos_{'_'.join([str(x) for x in camera_pos])}/"] = rollout_info

    wandb.log(all_rollout_info)

if __name__ == "__main__":
    app.run(main)
