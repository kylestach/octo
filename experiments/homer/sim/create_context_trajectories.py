import numpy as np
from absl import app, flags, logging
import glob
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)
flags.DEFINE_integer("num_traj", None, "Number of goals", required=True)
flags.DEFINE_string("accept_trajectory_key", None, "Success key", required=True)


def main(_):
    paths = glob.glob(os.path.join(FLAGS.data_path, "val/*.npy"))
    all_traj = [np.load(path, allow_pickle=True) for path in paths]
    all_traj = np.concatenate(all_traj, axis=0)
    success_traj = [traj for traj in all_traj if traj[-1]["observation"][f"info/{FLAGS.accept_trajectory_key}"]]
    success_traj = np.stack(success_traj)

    logging.info(f"Number of successful trajectories: {len(success_traj)}")
    indices = np.random.choice(range(len(success_traj)), size=FLAGS.num_traj, replace=False)
    context_traj = success_traj[indices]
    # convert array of dictionaries into dictionary of arrays
    context_traj = {k: np.array([[transition[k] for transition in traj] for traj in context_traj]) for k in context_traj[0, 0].keys()}
    context_traj["observation"] = {k: np.array([[transition[k] for transition in traj] for traj in context_traj["observation"]]) for k in context_traj["observation"][0, 0].keys()}

    np.save(os.path.join(FLAGS.data_path, "context_traj.npy"), context_traj)

if __name__ == "__main__":
    app.run(main)
