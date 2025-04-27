"""Open X-Embodiment Dataset Transforms

input: dict of features, each is batched, i.e. has leading time dimension
expected output:
step = {
    'observation': {
        <image_keys, depth_image_keys>
        state in chosen state representation
    },
    'action': action in chosen action representation,
    'language_instruction': str,
}
"""

from typing import Any, Dict

import tensorflow as tf

from octo.data.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_actions,
)

METRIC_WAYPOINT_SPACING = {
    "cory_hall": 0.06,
    "go_stanford": 0.12,
    "recon": 0.25,
    "sacson": 0.255,
    "scand": 0.38,
    "seattle": 0.35,
    "tartan_drive": 0.72,
}


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

def hard_bridge_eval_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

def ego4d_hamer_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ego4D Hamer hand detection data transform.
    Uses next N keypoint detections as targets & curates the data.
    Curation steps:
        1. Filter frames that do not detect the right hand in the current or next N-1 frames.
        2. Filter frames for which the **summed** delta-keypoint norm of the next N steps
           is above / below a threshold.
        3. Filter frames for which the **max** delta-keypoint norm of the next N steps
           is above a threshold (= usually jumps left-to-right hand).
    """
    N = 3  # Number of future steps for action targets
    DELTA_SUM_MIN = 20  # minimum summed delta-keypoint norm for N steps
    DELTA_SUM_MAX = 150  # minimum summed delta-keypoint norm for N steps
    DELTA_MAX = 100  # max delta-keypoint norm for each of the N steps
    STATE_ACTION_DIM = 7  # Action dimension of the padded actions -- make sure it's > 2*N

    assert STATE_ACTION_DIM >= 2 * N, "Need to choose an action dim that is larger than 2x number of keypoints"
    right_hand_kp = trajectory["action_dict"]["right"]["hand_center"]

    # Chunk detected keypoints into N step-chunks
    traj_len = tf.shape(right_hand_kp)[0]
    kp_chunk_indices = tf.broadcast_to(tf.range(N + 1)[None], [traj_len - N, N + 1]) + tf.broadcast_to(
        tf.range(traj_len - N)[:, None], [traj_len - N, N + 1]
    )
    chunked_kp = tf.gather(right_hand_kp, kp_chunk_indices)

    # Detect any chunk that does not have all keypoints detected (ie some are (0, 0))
    kp_norm = tf.linalg.norm(chunked_kp, axis=-1)
    missing_detection_mask = tf.reduce_any(tf.equal(kp_norm, 0.0), axis=1)

    # Compute delta keypoint actions
    chunked_delta_kp = chunked_kp[:, 1:] - chunked_kp[:, :-1]

    # Detect any chunks who's summed delta norms don't fall within the desired range
    chunked_delta_kp_norm = tf.linalg.norm(chunked_delta_kp, axis=-1)
    summed_delta_kp_norms = tf.reduce_sum(chunked_delta_kp_norm, axis=1)
    small_summed_delta_norm_mask = summed_delta_kp_norms < DELTA_SUM_MIN
    large_summed_delta_norm_mask = summed_delta_kp_norms > DELTA_SUM_MAX

    # Detect any chunks who's max delta norm is too large
    max_delta_kp_norms = tf.reduce_max(chunked_delta_kp_norm, axis=1)
    max_delta_norm_mask = max_delta_kp_norms > DELTA_MAX

    # Put all masks together for final filter
    total_mask = (
        ~missing_detection_mask & ~small_summed_delta_norm_mask & ~large_summed_delta_norm_mask & ~max_delta_norm_mask
    )

    # Don't use last N steps (since we don't have future keypoints for them)
    total_mask = tf.concat((total_mask, tf.zeros((N,), dtype=tf.bool)), axis=0)

    # Create padded versions of keypoint proprio and actions
    padded_proprio = tf.pad(right_hand_kp, [[0, 0], [0, STATE_ACTION_DIM - 2]])
    kp_action = tf.reshape(chunked_delta_kp, (traj_len - N, 2 * N))
    padded_kp_action = tf.pad(kp_action, [[0, N], [0, STATE_ACTION_DIM - 2 * N]])

    # Gather filtered transitions
    trajectory["observation"]["ego_image_1"] = tf.boolean_mask(trajectory["observation"]["ego_image_1"], total_mask)
    trajectory["observation"]["proprio"] = tf.boolean_mask(padded_proprio, total_mask)
    trajectory["action"] = tf.boolean_mask(padded_kp_action, total_mask)
    trajectory["language_instruction"] = tf.boolean_mask(trajectory["language_instruction"], total_mask)

    if "_traj_index" in trajectory:
        trajectory["_traj_index"] = tf.boolean_mask(trajectory["_traj_index"], total_mask)
    if "_frame_index" in trajectory:
        trajectory["_frame_index"] = tf.boolean_mask(trajectory["_frame_index"], total_mask)

    return trajectory

def fpha_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ego4D Hamer hand detection data transform.
    Uses next N keypoint detections as targets & curates the data.
    Curation steps:
        1. Filter frames that do not detect the right hand in the current or next N-1 frames.
        2. Filter frames for which the **summed** delta-keypoint norm of the next N steps
           is above / below a threshold.
        3. Filter frames for which the **max** delta-keypoint norm of the next N steps
           is above a threshold (= usually jumps left-to-right hand).
    """
    N = 1  # Number of future steps for action targets. some trajs are only 1 step long, so i made N=1
    DELTA_SUM_MIN = 20  # minimum summed delta-keypoint norm for N steps
    DELTA_SUM_MAX = 150  # minimum summed delta-keypoint norm for N steps
    DELTA_MAX = 100  # max delta-keypoint norm for each of the N steps
    STATE_ACTION_DIM = 7  # Action dimension of the padded actions -- make sure it's > 2*N

    assert STATE_ACTION_DIM >= 2 * N, "Need to choose an action dim that is larger than 2x number of keypoints"
    right_hand_kp = trajectory["action_dict"]["right"]["hand_center"]
    right_hand_kp *= 256  # Convert from normalized to pixel space

    # Chunk detected keypoints into N step-chunks
    traj_len = tf.shape(right_hand_kp)[0]

    kp_chunk_indices = tf.broadcast_to(tf.range(N + 1)[None], [traj_len - N, N + 1]) + tf.broadcast_to(
        tf.range(traj_len - N)[:, None], [traj_len - N, N + 1]
    )
    chunked_kp = tf.gather(right_hand_kp, kp_chunk_indices)

    # Detect any chunk that does not have all keypoints detected (ie some are (0, 0))
    kp_norm = tf.linalg.norm(chunked_kp, axis=-1)
    missing_detection_mask = tf.reduce_any(tf.equal(kp_norm, 0.0), axis=1)

    # Compute delta keypoint actions
    chunked_delta_kp = chunked_kp[:, 1:] - chunked_kp[:, :-1]

    # Detect any chunks who's summed delta norms don't fall within the desired range
    chunked_delta_kp_norm = tf.linalg.norm(chunked_delta_kp, axis=-1)
    summed_delta_kp_norms = tf.reduce_sum(chunked_delta_kp_norm, axis=1)
    small_summed_delta_norm_mask = summed_delta_kp_norms < DELTA_SUM_MIN
    large_summed_delta_norm_mask = summed_delta_kp_norms > DELTA_SUM_MAX

    # Detect any chunks who's max delta norm is too large
    max_delta_kp_norms = tf.reduce_max(chunked_delta_kp_norm, axis=1)
    max_delta_norm_mask = max_delta_kp_norms > DELTA_MAX

    # Put all masks together for final filter
    total_mask = (
            ~missing_detection_mask & ~small_summed_delta_norm_mask & ~large_summed_delta_norm_mask & ~max_delta_norm_mask
    )

    # Don't use last N steps (since we don't have future keypoints for them)
    total_mask = tf.concat((total_mask, tf.zeros((N,), dtype=tf.bool)), axis=0)
    kp_action = tf.reshape(chunked_delta_kp, (traj_len - N, 2 * N))
    padded_kp_action = tf.pad(kp_action, [[0, N], [0, STATE_ACTION_DIM - 2 * N]])

    # Create padded versions of keypoint proprio and actions
    padded_proprio = tf.pad(right_hand_kp, [[0, 0], [0, STATE_ACTION_DIM - 2]])

    # Gather filtered transitions
    trajectory["observation"]["ego_image_1"] = tf.boolean_mask(trajectory["observation"]["ego_image_1"], total_mask)
    trajectory["observation"]["proprio"] = tf.boolean_mask(padded_proprio, total_mask)
    trajectory["action"] = tf.boolean_mask(padded_kp_action, total_mask)
    trajectory["language_instruction"] = tf.boolean_mask(trajectory['traj_metadata']['episode_metadata']['narration'], total_mask)

    if "_traj_index" in trajectory:
        trajectory["_traj_index"] = tf.boolean_mask(trajectory["_traj_index"], total_mask)
    if "_frame_index" in trajectory:
        trajectory["_frame_index"] = tf.boolean_mask(trajectory["_frame_index"], total_mask)

    return trajectory

def h2o_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ego4D Hamer hand detection data transform.
    Uses next N keypoint detections as targets & curates the data.
    Curation steps:
        1. Filter frames that do not detect the right hand in the current or next N-1 frames.
        2. Filter frames for which the **summed** delta-keypoint norm of the next N steps
           is above / below a threshold.
        3. Filter frames for which the **max** delta-keypoint norm of the next N steps
           is above a threshold (= usually jumps left-to-right hand).
    """
    N = 3  # Number of future steps for action targets
    DELTA_SUM_MIN = 20  # minimum summed delta-keypoint norm for N steps
    DELTA_SUM_MAX = 150  # minimum summed delta-keypoint norm for N steps
    DELTA_MAX = 100  # max delta-keypoint norm for each of the N steps
    STATE_ACTION_DIM = 7  # Action dimension of the padded actions -- make sure it's > 2*N

    assert STATE_ACTION_DIM >= 2 * N, "Need to choose an action dim that is larger than 2x number of keypoints"
    right_hand_kp = trajectory["action_dict"]["right"]["hand_center"]
    right_hand_kp *= 256  # Convert from normalized to pixel space
    # Chunk detected keypoints into N step-chunks
    traj_len = tf.shape(right_hand_kp)[0]
    kp_chunk_indices = tf.broadcast_to(tf.range(N + 1)[None], [traj_len - N, N + 1]) + tf.broadcast_to(
        tf.range(traj_len - N)[:, None], [traj_len - N, N + 1]
    )
    chunked_kp = tf.gather(right_hand_kp, kp_chunk_indices)

    # Detect any chunk that does not have all keypoints detected (ie some are (0, 0))
    kp_norm = tf.linalg.norm(chunked_kp, axis=-1)
    missing_detection_mask = tf.reduce_any(tf.equal(kp_norm, 0.0), axis=1)

    # Compute delta keypoint actions
    chunked_delta_kp = chunked_kp[:, 1:] - chunked_kp[:, :-1]

    # Detect any chunks who's summed delta norms don't fall within the desired range
    chunked_delta_kp_norm = tf.linalg.norm(chunked_delta_kp, axis=-1)
    summed_delta_kp_norms = tf.reduce_sum(chunked_delta_kp_norm, axis=1)
    small_summed_delta_norm_mask = summed_delta_kp_norms < DELTA_SUM_MIN
    large_summed_delta_norm_mask = summed_delta_kp_norms > DELTA_SUM_MAX

    # Detect any chunks who's max delta norm is too large
    max_delta_kp_norms = tf.reduce_max(chunked_delta_kp_norm, axis=1)
    max_delta_norm_mask = max_delta_kp_norms > DELTA_MAX

    # Put all masks together for final filter
    total_mask = (
        ~missing_detection_mask & ~small_summed_delta_norm_mask & ~large_summed_delta_norm_mask & ~max_delta_norm_mask
    )

    # Don't use last N steps (since we don't have future keypoints for them)
    total_mask = tf.concat((total_mask, tf.zeros((N,), dtype=tf.bool)), axis=0)

    # Create padded versions of keypoint proprio and actions
    padded_proprio = tf.pad(right_hand_kp, [[0, 0], [0, STATE_ACTION_DIM - 2]])
    kp_action = tf.reshape(chunked_delta_kp, (traj_len - N, 2 * N))
    padded_kp_action = tf.pad(kp_action, [[0, N], [0, STATE_ACTION_DIM - 2 * N]])

    # Gather filtered transitions
    trajectory["observation"]["ego_image_1"] = tf.boolean_mask(trajectory["observation"]["ego_image_1"], total_mask)
    trajectory["observation"]["proprio"] = tf.boolean_mask(padded_proprio, total_mask)
    trajectory["action"] = tf.boolean_mask(padded_kp_action, total_mask)
    trajectory["language_instruction"] = tf.boolean_mask(trajectory['traj_metadata']['episode_metadata']['narration'], total_mask)

    if "_traj_index" in trajectory:
        trajectory["_traj_index"] = tf.boolean_mask(trajectory["_traj_index"], total_mask)
    if "_frame_index" in trajectory:
        trajectory["_frame_index"] = tf.boolean_mask(trajectory["_frame_index"], total_mask)

    return trajectory

def ssv2_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ego4D Hamer hand detection data transform.
    Uses next N keypoint detections as targets & curates the data.
    Curation steps:
        1. Filter frames that do not detect the right hand in the current or next N-1 frames.
        2. Filter frames for which the **summed** delta-keypoint norm of the next N steps
           is above / below a threshold.
        3. Filter frames for which the **max** delta-keypoint norm of the next N steps
           is above a threshold (= usually jumps left-to-right hand).
    """
    N = 3  # Number of future steps for action targets
    DELTA_SUM_MIN = 20  # minimum summed delta-keypoint norm for N steps
    DELTA_SUM_MAX = 150  # minimum summed delta-keypoint norm for N steps
    DELTA_MAX = 100  # max delta-keypoint norm for each of the N steps
    STATE_ACTION_DIM = 7  # Action dimension of the padded actions -- make sure it's > 2*N

    assert STATE_ACTION_DIM >= 2 * N, "Need to choose an action dim that is larger than 2x number of keypoints"
    right_hand_kp = trajectory["action_dict"]["right"]["hand_center"]
    right_hand_kp *= 256  # Convert from normalized to pixel space
    # Chunk detected keypoints into N step-chunks
    traj_len = tf.shape(right_hand_kp)[0]
    kp_chunk_indices = tf.broadcast_to(tf.range(N + 1)[None], [traj_len - N, N + 1]) + tf.broadcast_to(
        tf.range(traj_len - N)[:, None], [traj_len - N, N + 1]
    )
    chunked_kp = tf.gather(right_hand_kp, kp_chunk_indices)

    # Detect any chunk that does not have all keypoints detected (ie some are (0, 0))
    kp_norm = tf.linalg.norm(chunked_kp, axis=-1)
    missing_detection_mask = tf.reduce_any(tf.equal(kp_norm, 0.0), axis=1)

    # Compute delta keypoint actions
    chunked_delta_kp = chunked_kp[:, 1:] - chunked_kp[:, :-1]

    # Detect any chunks who's summed delta norms don't fall within the desired range
    chunked_delta_kp_norm = tf.linalg.norm(chunked_delta_kp, axis=-1)
    summed_delta_kp_norms = tf.reduce_sum(chunked_delta_kp_norm, axis=1)
    small_summed_delta_norm_mask = summed_delta_kp_norms < DELTA_SUM_MIN
    large_summed_delta_norm_mask = summed_delta_kp_norms > DELTA_SUM_MAX

    # Detect any chunks who's max delta norm is too large
    max_delta_kp_norms = tf.reduce_max(chunked_delta_kp_norm, axis=1)
    max_delta_norm_mask = max_delta_kp_norms > DELTA_MAX

    # Put all masks together for final filter
    total_mask = (
        ~missing_detection_mask & ~small_summed_delta_norm_mask & ~large_summed_delta_norm_mask & ~max_delta_norm_mask
    )

    # Don't use last N steps (since we don't have future keypoints for them)
    total_mask = tf.concat((total_mask, tf.zeros((N,), dtype=tf.bool)), axis=0)

    # Create padded versions of keypoint proprio and actions
    padded_proprio = tf.pad(right_hand_kp, [[0, 0], [0, STATE_ACTION_DIM - 2]])
    kp_action = tf.reshape(chunked_delta_kp, (traj_len - N, 2 * N))
    padded_kp_action = tf.pad(kp_action, [[0, N], [0, STATE_ACTION_DIM - 2 * N]])

    # Gather filtered transitions
    trajectory["observation"]["ego_image_1"] = tf.boolean_mask(trajectory["observation"]["ego_image_1"], total_mask)
    trajectory["observation"]["proprio"] = tf.boolean_mask(padded_proprio, total_mask)
    trajectory["action"] = tf.boolean_mask(padded_kp_action, total_mask)
    trajectory["language_instruction"] = tf.boolean_mask(trajectory['traj_metadata']['episode_metadata']['label'], total_mask)

    if "_traj_index" in trajectory:
        trajectory["_traj_index"] = tf.boolean_mask(trajectory["_traj_index"], total_mask)
    if "_frame_index" in trajectory:
        trajectory["_frame_index"] = tf.boolean_mask(trajectory["_frame_index"], total_mask)

    return trajectory

def epic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ego4D Hamer hand detection data transform.
    Uses next N keypoint detections as targets & curates the data.
    Curation steps:
        1. Filter frames that do not detect the right hand in the current or next N-1 frames.
        2. Filter frames for which the **summed** delta-keypoint norm of the next N steps
           is above / below a threshold.
        3. Filter frames for which the **max** delta-keypoint norm of the next N steps
           is above a threshold (= usually jumps left-to-right hand).
    """
    N = 2  # Number of future steps for action targets; changed bc some trajs are smaller than 3
    DELTA_SUM_MIN = 20  # minimum summed delta-keypoint norm for N steps
    DELTA_SUM_MAX = 150  # minimum summed delta-keypoint norm for N steps
    DELTA_MAX = 100  # max delta-keypoint norm for each of the N steps
    STATE_ACTION_DIM = 7  # Action dimension of the padded actions -- make sure it's > 2*N

    assert STATE_ACTION_DIM >= 2 * N, "Need to choose an action dim that is larger than 2x number of keypoints"
    right_hand_kp = trajectory["action_dict"]["right"]["hand_center"]
    right_hand_kp *= 256  # Convert from normalized to pixel space

    # Chunk detected keypoints into N step-chunks
    traj_len = tf.shape(right_hand_kp)[0]
    kp_chunk_indices = tf.broadcast_to(tf.range(N + 1)[None], [traj_len - N, N + 1]) + tf.broadcast_to(
        tf.range(traj_len - N)[:, None], [traj_len - N, N + 1]
    )
    chunked_kp = tf.gather(right_hand_kp, kp_chunk_indices)

    # Detect any chunk that does not have all keypoints detected (ie some are (0, 0))
    kp_norm = tf.linalg.norm(chunked_kp, axis=-1)
    missing_detection_mask = tf.reduce_any(tf.equal(kp_norm, 0.0), axis=1)

    # Compute delta keypoint actions
    chunked_delta_kp = chunked_kp[:, 1:] - chunked_kp[:, :-1]

    # Detect any chunks who's summed delta norms don't fall within the desired range
    chunked_delta_kp_norm = tf.linalg.norm(chunked_delta_kp, axis=-1)
    summed_delta_kp_norms = tf.reduce_sum(chunked_delta_kp_norm, axis=1)
    small_summed_delta_norm_mask = summed_delta_kp_norms < DELTA_SUM_MIN
    large_summed_delta_norm_mask = summed_delta_kp_norms > DELTA_SUM_MAX

    # Detect any chunks who's max delta norm is too large
    max_delta_kp_norms = tf.reduce_max(chunked_delta_kp_norm, axis=1)
    max_delta_norm_mask = max_delta_kp_norms > DELTA_MAX

    # Put all masks together for final filter
    total_mask = (
        ~missing_detection_mask & ~small_summed_delta_norm_mask & ~large_summed_delta_norm_mask & ~max_delta_norm_mask
    )

    # Don't use last N steps (since we don't have future keypoints for them)
    total_mask = tf.concat((total_mask, tf.zeros((N,), dtype=tf.bool)), axis=0)
    kp_action = tf.reshape(chunked_delta_kp, (traj_len - N, 2 * N))
    padded_kp_action = tf.pad(kp_action, [[0, N], [0, STATE_ACTION_DIM - 2 * N]])

    # Create padded versions of keypoint proprio and actions
    padded_proprio = tf.pad(right_hand_kp, [[0, 0], [0, STATE_ACTION_DIM - 2]])

    # Gather filtered transitions
    trajectory["observation"]["ego_image_1"] = tf.boolean_mask(trajectory["observation"]["ego_image_1"], total_mask)
    trajectory["observation"]["proprio"] = tf.boolean_mask(padded_proprio, total_mask)
    trajectory["action"] = tf.boolean_mask(padded_kp_action, total_mask)
    trajectory["language_instruction"] = tf.boolean_mask(trajectory['traj_metadata']["language_instruction"], total_mask)

    if "_traj_index" in trajectory:
        trajectory["_traj_index"] = tf.boolean_mask(trajectory["_traj_index"], total_mask)
    if "_frame_index" in trajectory:
        trajectory["_frame_index"] = tf.boolean_mask(trajectory["_frame_index"], total_mask)

    return trajectory

def epic_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # check if the hands are present
    trajectory["observation"]["tcp_point_3d_left"] = tf.where(
        trajectory["observation"]["has_hand_left"][:, tf.newaxis],
        trajectory["observation"]["tcp_point_3d_left"],
        tf.zeros_like(trajectory["observation"]["tcp_point_3d_left"]),
    )

    # Use tf.where for right hand as well
    trajectory["observation"]["tcp_point_3d_right"] = tf.where(
        trajectory["observation"]["has_hand_right"][:, tf.newaxis],
        trajectory["observation"]["tcp_point_3d_right"],
        tf.zeros_like(trajectory["observation"]["tcp_point_3d_right"]),
    )

    concatenated_tcp = tf.concat(
        [
            trajectory["observation"]["tcp_point_3d_left"],
            trajectory["observation"]["tcp_point_3d_right"],
        ],
        axis=1,
    )
    # compute relative actions across time dimension
    relative_actions = tf.experimental.numpy.diff(concatenated_tcp, axis=0)
    # add zero padding to make the shape consistent
    zero_padding = tf.zeros_like(relative_actions[0:1])
    relative_actions = tf.concat([relative_actions, zero_padding], axis=0)
    # Add an extra zero column to make the shape (timestep, 7), so we can mix with 7-DoF datasets
    extra_zero_column = tf.zeros_like(relative_actions[:, :1])
    relative_actions = tf.concat([relative_actions, extra_zero_column], axis=1)
    trajectory["action"] = relative_actions
    return trajectory

def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    gripper_value = tf.io.decode_compressed(
        trajectory["observation"]["gripper_closed"], compression_type="ZLIB"
    )
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["proprio"] = tf.concat(
        (
            tf.reshape(eef_value, (-1, 7)),
            tf.reshape(gripper_value, (-1, 1)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def taco_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["robot_obs"][:, :6],
            trajectory["observation"]["robot_obs"][:, 7:8],
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def berkeley_cable_routing_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1)
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_states"],
            trajectory["observation"]["gripper_states"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_autolab_ur5_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = trajectory["observation"].pop(
        "image_with_depth"
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"][
        :, 6:14
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "effector_translation"
    ]
    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding="UTF-8"
    )
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[
        :, :1
    ].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][
        ..., 0
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["ee_position"],
            trajectory["observation"]["ee_orientation"],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :7]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            trajectory["observation"]["state"][:, -3:-2],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(
        trajectory["observation"]["depth"][..., 0], tf.float32
    )
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -6:]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_pose"],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :7]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["present/xyz"],
            trajectory["observation"]["present/axis_angle"],
            trajectory["observation"]["present/sensed_close"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_pose"
    ]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
            trajectory["observation"]["state"][:, -1:],
        ),
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["pose"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # invert gripper
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :-1],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=1,
    )

    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 30Hz to 10Hz
    factor = 3
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_pos"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # recompute actions for downsampled sequence
    joint_actions = (
        trajectory["observation"]["joint_pos"][1:, :7]
        - trajectory["observation"]["joint_pos"][:-1, :7]
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)

    # recombine to get full actions, invert gripper
    traj_truncated["action"] = tf.concat(
        [joint_actions, invert_gripper_actions(trajectory["action"][:-1, -1:])],
        axis=1,
    )

    return traj_truncated


def kaist_nonprehensible_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -7:]
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
            trajectory["observation"]["end_effector_pose"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["ground_truth_states"]["EE"],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, 6:7],
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def omnimimic_gnm_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    traj_len = tf.shape(trajectory["action"])[0]
    action_horizon = 100

    # Pad trajectory states
    padding = tf.tile(trajectory["observation"]["state"][-1:, :], [action_horizon, 1])
    trajectory["observation"]["state"] = tf.concat(
        (trajectory["observation"]["state"], padding), axis=0
    )

    # Get next len_seq_pred indices
    indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(1, action_horizon + 1)
    global_waypoints = tf.gather(trajectory["observation"]["state"], indices)[:, :, :2]

    # Get current position indices
    curr_pos_indices = tf.reshape(tf.range(traj_len), [-1, 1]) + tf.range(
        0, action_horizon
    )
    curr_pos = tf.gather(trajectory["observation"]["state"], curr_pos_indices)[
        :, :, :2
    ]  # delta waypoints

    global_waypoints -= curr_pos
    global_waypoints = tf.expand_dims(global_waypoints, 2)
    actions = tf.squeeze(
        tf.linalg.matmul(
            global_waypoints,
            tf.expand_dims(trajectory["observation"]["yaw_rotmat"][:, :2, :2], 1),
        ),
        2,
    )

    normalization_factor = 1.0
    for dataset_name, value in METRIC_WAYPOINT_SPACING.items():
        if tf.strings.regex_full_match(
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0],
            f".*{dataset_name}.*",
        ):
            normalization_factor = value
    normalization_factor = tf.cast(normalization_factor, tf.float64)
    actions = actions / normalization_factor

    trajectory["action"] = actions

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    return trajectory


def old_gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    traj_len = tf.shape(trajectory["action"])[0]
    action_horizon = 4

    # compute rot matrix
    yaw = trajectory["observation"]["yaw"]
    rot_mat = tf.convert_to_tensor(
        [
            [tf.cos(yaw), -tf.sin(yaw)],
            [tf.sin(yaw), tf.cos(yaw)],
        ]
    )
    rot_mat = tf.transpose(rot_mat, [3, 2, 0, 1])[0]

    # chunk actions and recompute as relative to the start of the chunk
    pos = trajectory["observation"]["position"]
    start = tf.broadcast_to(pos[:, None], [traj_len, action_horizon, 2])
    end_indices = tf.range(traj_len)[:, None] + tf.range(1, action_horizon + 1)
    end_indices = tf.minimum(end_indices, traj_len - 1)
    end = tf.gather(pos, end_indices)
    delta = end - start
    action = tf.matmul(delta[:, :, None], rot_mat[:, None])[:, :, 0]  # * scaling_factor

    # get normalization factor
    normalization_factor = 1.0
    for dataset_name, value in METRIC_WAYPOINT_SPACING.items():
        if tf.strings.regex_full_match(
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0],
            f".*{dataset_name}.*",
        ):
            normalization_factor = value
    normalization_factor = tf.cast(normalization_factor, tf.float64)
    action = action / normalization_factor

    trajectory["action"] = action

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    return trajectory


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 50Hz to 10Hz
    factor = 5
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def mujoco_manip_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    gripper_action = invert_gripper_actions(trajectory["action"][:, -1:] / 255)
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :6], gripper_action), axis=-1
    )
    return trajectory


def go1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def aloha_pen_uncap_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def aloha_dough_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["language_instruction"] = trajectory["global_instruction"]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

def aloha_pick_place_full_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory

def aria_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory

def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_dict"]["cartesian_velocity"],
            invert_gripper_actions(trajectory["action_dict"]["gripper_position"]),
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory

# fake droid!! just demos we collected
def ria_droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory

def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    eef_state = trajectory["observation"]["state"][:, :6]
    gripper_state = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    
    trajectory["observation"]["proprio"] = tf.concat(
        (
            eef_state,
            gripper_state,
        ),
        axis=-1,
    )
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "hand_epic_dataset": epic_dataset_transform,
    "ego4d_hamer": ego4d_hamer_transform,
    "epic_kitchens": epic_transform,
    "h2_o_dataset": h2o_transform,
    "fpha_dataset": fpha_transform,
    "ss_v2_dataset": ssv2_transform,
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_dataset_transform,
    "taco_extra": taco_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "omnimimic_gnm_dataset": omnimimic_gnm_transform,
    "aloha_dagger_dataset": aloha_dataset_transform,
    "aloha_mobile_dataset": aloha_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    "mujoco_manip": mujoco_manip_dataset_transform,
    "go1": go1_dataset_transform,
    "aloha_pen_uncap_diverse_dataset": aloha_pen_uncap_dataset_transform,
    "aloha_dough_cut_dataset": aloha_dough_dataset_transform,
    "aloha_lucy_dataset": aloha_dough_dataset_transform,
    "aloha_drawer_dataset": aloha_dough_dataset_transform,
    "aloha_pick_place_dataset": aloha_dough_dataset_transform,
    "aloha_pick_place_full_dataset": aloha_pick_place_full_dataset_transform,
    "aloha_bread_dataset": aloha_pick_place_full_dataset_transform,
    "aloha_spoons_in_bowls_dataset": aloha_pick_place_full_dataset_transform,
    "aloha_wipe_plate": aloha_pick_place_full_dataset_transform,
    "aloha_long_horizon_dataset": aloha_pick_place_full_dataset_transform,
    "aloha_static_dataset": aloha_dough_dataset_transform,
    "aloha_sushi_cut_full_dataset": aloha_dough_dataset_transform,
    "droid": droid_dataset_transform,
    "droid_wipe": droid_dataset_transform,
    "libero_90": libero_dataset_transform,
    "hard_bridge_eval": hard_bridge_eval_transform,
    "aria_dataset": aria_dataset_transform,
    "droid_dataset": ria_droid_dataset_transform, # fake droid!! just demos we collected

}
