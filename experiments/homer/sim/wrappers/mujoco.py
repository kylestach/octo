import gym
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.homer.sim.wrappers.dmcgym import DMCGYM
import mujoco_manipulation
import tensorflow as tf
from typing import Sequence, Optional
import tensorflow_graphics.geometry.transformation.rotation_matrix_3d as tfr

def convert_obs(obs):
    obs["image_primary"] = obs.pop("pixels")
    obs["proprio"] = np.concatenate(
        [
            obs.pop("end_effector_pos"),
            Rotation.from_quat(obs.pop("end_effector_quat")).as_euler("xyz"),
            obs.pop("right_finger_qpos"),
        ]
    )
    obs.pop("left_finger_qpos")
    return obs


def filter_info_keys(info):
    keep_keys = ["place_success"]
    return {k: v for k, v in info.items() if k in keep_keys}

def standardize_action(action):
    # standardization transform
    gripper_action = 1 - (action[:, -1:] / 255)
    return np.concatenate([action[:, :6], gripper_action], axis=1)

def unstandardize_action(action):
    # reverse standardization transform
    gripper_action = (1 - action[:, -1:]) * 255
    return np.concatenate([action[:, :6], gripper_action], axis=1)

def normalize_action(action, normalization_statistics):
    mask = normalization_statistics.get(
        "mask", np.ones_like(normalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = np.where(
        mask,
        (action - normalization_statistics["mean"]) / (normalization_statistics["std"] + 1e-8),
        action,
    )
    return action

def unnormalize_action(action, unnormalization_statistics):
    mask = unnormalization_statistics.get(
        "mask", np.ones_like(unnormalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = np.where(
        mask,
        (action * unnormalization_statistics["std"]) + unnormalization_statistics["mean"],
        action,
    )
    return action

def rotate(rot_matrix, action):
    rotated_delta = tf.matmul(rot_matrix, action[..., :3, None])[..., 0]
    return tf.concat([rotated_delta, action[..., 3:]], axis=-1)

def unrotate_action(action, action_rot):
    # reverse rotation augmentation
    action = tf.cast(action, tf.float32)
    action_rot = tf.cast(action_rot, tf.float32)
    rot_matrix = tfr.inverse(tfr.from_euler(action_rot))
    rotated_delta = rotate(rot_matrix[None], action[:, :3]).numpy()
    return np.concatenate([rotated_delta, action[:, 3:]], axis=1)

def rotate_action(action, action_rot):
    # rotation augmentation
    action = tf.cast(action, tf.float32)
    action_rot = tf.cast(action_rot, tf.float32)
    rot_matrix = tfr.from_euler(action_rot)
    rotated_delta = rotate(rot_matrix[None], action[:, :3]).numpy()
    return np.concatenate([rotated_delta, action[:, 3:]], axis=1)

class MujocoManipWrapper(gym.Wrapper):
    """
    Wrapper for Mujoco Manipulation sim environments.
    """

    def __init__(self, env: gym.Env, goals_path: str, unnormalization_statistics: dict, context_path: Optional[str] = None, action_rot: Optional[Sequence[float]] = None):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((128, 128, 3)),
                    high=255 * np.ones((128, 128, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.zeros((7,)), high=np.ones((7,)), dtype=np.uint8
                ),
            }
        )
        self.current_goal = None
        self.place_success = False
        self.unnormalization_statistics = unnormalization_statistics
        self.action_rot = action_rot
        with tf.io.gfile.GFile(tf.io.gfile.join(goals_path), "rb") as f:
            self.goals = np.load(f, allow_pickle=True).item()
        if context_path is not None:
            with tf.io.gfile.GFile(tf.io.gfile.join(context_path), "rb") as f:
                self.context = np.load(f, allow_pickle=True).item()

    def step(self, action, *args):
        if self.action_rot is not None:
            action = unrotate_action(action[None], self.action_rot)[0]
        action = unnormalize_action(action[None], self.unnormalization_statistics)[0]
        action = unstandardize_action(action[None])[0]
        obs, reward, done, trunc, info = self.env.step(action, *args)
        info = filter_info_keys(info)
        self.place_success = info["place_success"]
        return (
            convert_obs(obs),
            reward,
            done,
            trunc,
            info,
        )

    def get_goal(self):
        return self.current_goal

    def get_instruction(self):
        return "put the shoe on the circle"

    def get_context(self, context_window_size):
        idx = np.random.randint(len(self.context["observation"]["image"]))
        context = dict()
        context["observation"] = dict()
        context["observation"]["image_primary"] = self.context["observation"]["image"][idx]
        context["action"] = standardize_action(self.context["action"][idx])
        context["action"] = normalize_action(context["action"], self.unnormalization_statistics)
        if self.action_rot is not None:
            context["action"] = rotate_action(context["action"], self.action_rot)

        start_idx = np.random.randint(max(len(context["action"]) - context_window_size, 0))
        context_indices = np.arange(start_idx, start_idx + context_window_size)
        timestep_pad_mask = context_indices <= len(context["action"]) - 1
        context["observation"]["image_primary"] = context["observation"]["image_primary"][context_indices][None]
        context["action"] = context["action"][context_indices][None]
        context["observation"]["timestep_pad_mask"] = timestep_pad_mask[None]

        return context


    def get_episode_metrics(self):
        return {"place_success": self.place_success}

    def reset(self, **kwargs):
        idx = np.random.randint(len(self.goals["observation"]["image"]))
        goal_image = self.goals["observation"]["image"][idx]
        original_object_positions = self.goals["observation"]["info/initial_positions"][
            idx
        ]
        # original_object_quats = self.goals["observation"]["info/initial_quats"][idx]
        target_position = self.goals["observation"]["info/target_position"][idx]
        object_names = self.goals["observation"]["info/object_names"][idx]
        target_object = self.goals["observation"]["info/target_object"][idx]
        self.env.task.change_props(object_names)
        self.env.task.init_prop_poses = original_object_positions
        self.env.task.target_pos = target_position
        self.env.target_obj = target_object
        obs, info = self.env.reset()
        obs = convert_obs(obs)

        goal = {"image_primary": goal_image[None]}

        self.current_goal = goal
        self.place_success = False

        info = filter_info_keys(info)
        return obs, info

# register gym environment
gym.register(
    "franka-shoe-pick-place",
    entry_point=lambda base_env_kwargs={}, **kwargs: MujocoManipWrapper(DMCGYM(
        mujoco_manipulation.load("franka_shoe_pick_and_place", **base_env_kwargs)), **kwargs
    ),
)
