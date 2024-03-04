"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import cv2
import jax
import pickle
import numpy as np
import yaml
import robots
import gym
import tensorflow as tf
import dlimp as dl
import json
from scipy.spatial.transform import Rotation as R

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)
tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)
flags.DEFINE_string("robot_config_path", "/scr/jhejna/robot-lightning/configs/iliad_franka_orca_abs.yaml", "Path to robot config")
flags.DEFINE_string("video_save_path", "/scr/jhejna/videos/1_30/", "Path to save video")
flags.DEFINE_string("instr", None, "langauge instruction.")
flags.DEFINE_string("goal_image", None, "Path to goal image.")
flags.DEFINE_integer("num_timesteps", 150, "num timesteps")
flags.DEFINE_integer("horizon", 2, "Observation history length")
flags.DEFINE_integer("exec_horizon", 2, "Length of action sequence to execute")
flags.DEFINE_bool("deterministic", False, "Whether or not to sample actions deterministically.")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions.")

def convert_obs(obs):
    # We need to concatenate the state observations, then change the image observations
    new_obs = dict()
    new_obs["image_primary"] = obs["agent_image"]
    # For consistency resize using dlimp
    new_obs["image_wrist"] = dl.transforms.resize_image(obs["wrist_image"], size=(128, 128)).numpy()
    new_obs["proprio"] = np.concatenate((obs["state"]["ee_pos"], obs["state"]["ee_quat"], obs["state"]["gripper_pos"],))
    return new_obs

def rmat_to_euler_scipy(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat_scipy(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def R6_to_rmat_scipy(r6_mat):
    a1, a2 = r6_mat[:3], r6_mat[3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.vstack((b1, b2, b3))
    return out


class ILIADWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        # We need to convert the action back to absolute position
        ee_pos = action[:3]
        ee_rot6d = action[3:9]
        gripper = action[-1:]
        euler = rmat_to_euler_scipy(R6_to_rmat_scipy(ee_rot6d))
        action = np.concatenate([ee_pos, euler, gripper], axis=-1, dtype=np.float32)

        action[-1] = 1 - action[-1]
        print(action)

        obs, reward, done, truncated, info = self.env.step(action)
        return convert_obs(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return convert_obs(obs), info

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return vec / norm

def rot6d_to_euler(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    matrix = np.stack((b1, b2, b3), axis=-2)

    quat = R.from_matrix(matrix).as_quat()
    quat = np.roll(quat, 1, axis=0)
    
    return R.from_quat(quat).as_euler("xyz")


def main(_):
    # set up the franka client
    with open(FLAGS.robot_config_path, "r") as f:
        robot_config = yaml.load(f, Loader=yaml.Loader)
    env = robots.RobotEnv(**robot_config)
    env = ILIADWrapper(env)

    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    env = HistoryWrapper(env, FLAGS.horizon)
    env = RHCWrapper(env, FLAGS.exec_horizon)

    # create policy functions
    @jax.jit
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng,
        argmax=False,
        temperature=1.0
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            argmax=argmax,
            temperature=temperature,
        )
        action_metadata = pretrained_model.dataset_statistics['action']

        # action_loc = action_metadata["mean"]
        # action_scale = action_metadata["std"]

        ac_min = action_metadata["p01"]
        ac_max = action_metadata["p99"]
        action_loc = 0.5 * (ac_max + ac_min)
        action_scale = 0.5 * (ac_max - ac_min)

        mask = action_metadata.get("mask", jax.numpy.ones_like(action_metadata["mean"], dtype=bool))
        actions = actions[..., : len(mask)]
        actions = jax.numpy.where(mask, (actions * action_scale) + action_loc, actions)

        # remove batch dim
        return actions[0]
        
    def supply_rng(f, rng=jax.random.PRNGKey(0)):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return f(*args, rng=key, **kwargs)
        return wrapped

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            argmax=FLAGS.deterministic,
            temperature=FLAGS.temperature,
        )
    )

    # Construct the goals
    assert FLAGS.instr is not None or FLAGS.goal_image is not None
    # Setup the goal.
    if FLAGS.instr is not None:
        texts = [FLAGS.instr]
    else:
        texts = None

    if FLAGS.goal_image is not None:
        with open(FLAGS.goal_image, 'rb') as f:
            goals = pickle.load(f)
        goals = convert_obs(goals)
        goals = jax.tree_map(lambda x: x[None], goals)
    else:
        goals = None

    task = model.create_tasks(goals=goals, texts=texts)

    # goal sampling loop
    while True:
        input("Press [Enter] to start.")

        obs, _ = env.reset()
        time.sleep(2.0)
        terminated = False

        # do rollout
        images = [obs["image_primary"]]
        t = 0

        try:
            while t < FLAGS.num_timesteps and not terminated:

                # get action
                action = np.array(policy_fn(obs, task))

                # perform environment step
                obs, _, _, truncated, _ = env.step(action)
                images.append(obs["image_primary"])
                t += 1

                if truncated:
                    break
        except KeyboardInterrupt:
            print("ending early!")

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            # Create the directory for this trajectory
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            user_input = input("Was success or discard? [y/n/d]")
            if user_input == "d":
                continue
            elif user_input == "y":
                success = True
            else:
                success = False

            ckpt_tag = os.path.basename(os.path.normpath(FLAGS.checkpoint_weights_path)) + "_" + str(FLAGS.checkpoint_step)
            checkpoint_eval_path = os.path.join(FLAGS.video_save_path, ckpt_tag)
            os.makedirs(checkpoint_eval_path, exist_ok=True)
            save_path = os.path.join(
                checkpoint_eval_path,
                f"{curr_time}_success-{success}.mp4",
            )
            video = np.stack(images)[:, -1, :, :, :]
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (256, 256))
            for img in video:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()

if __name__ == "__main__":
    app.run(main)