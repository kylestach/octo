import tensorflow as tf
from typing import Any, Dict
from octo.data.utils.data_utils import invert_gripper_actions
import tensorflow_graphics.geometry.transformation as tfg
import numpy as np


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)

def quat_to_rmat(quat):
    return tfg.rotation_matrix_3d.from_quaternion(quat)

def rmat_to_rot6d(rmat):
    r6 = rmat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat

def iliad_franka_dataset_transform_rel(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"]["ee_pos"],
            trajectory["observation"]["state"]["ee_quat"],
            trajectory["observation"]["state"]["gripper_pos"],
        ),
        axis=1,
    )
    gripper_action = 0.5 * (trajectory["action"][:, -1:] + 1) # Range 0 to 1
    gripper_action = invert_gripper_actions(gripper_action)
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :-1], gripper_action), axis=-1
    )
    return trajectory

def iliad_franka_dataset_transform_rel_r6(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"]["ee_pos"],
            trajectory["observation"]["state"]["ee_quat"],
            trajectory["observation"]["state"]["gripper_pos"],
        ),
        axis=1,
    )
    gripper_action = 0.5 * (trajectory["action"][:, -1:] + 1) # Range 0 to 1
    gripper_action = invert_gripper_actions(gripper_action)
    delta_action = trajectory["action"][:, :3]
    delta_rot = rmat_to_rot6d(euler_to_rmat(trajectory["action"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (delta_action, delta_rot, gripper_action), axis=-1
    )
    return trajectory

def iliad_franka_dataset_transform_abs(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    low = np.array([-0.05, -0.05, -0.05, -0.2, -0.2, -0.2, 0])
    high = np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1])
    action = trajectory["action"]
    unscaled_action = low + (0.5 * (action + 1.0) * (high - low))
    delta_pos, delta_ori = unscaled_action[..., :3], unscaled_action[..., 3:6]
    gripper_action = unscaled_action[..., -1:]
    ee_pos = trajectory["observation"]["state"]["ee_pos"]
    ee_quat = trajectory["observation"]["state"]["ee_quat"]
    new_pos = ee_pos + delta_pos
    rmat = euler_to_rmat(delta_ori) * quat_to_rmat(ee_quat)
    rot6d = rmat_to_rot6d(rmat)

    trajectory["action"] = tf.concat(
        (
            new_pos,
            rot6d,
            invert_gripper_actions(gripper_action),
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"]["ee_pos"],
            trajectory["observation"]["state"]["ee_quat"],
            trajectory["observation"]["state"]["gripper_pos"],
        ),
        axis=1,
    )

    return trajectory