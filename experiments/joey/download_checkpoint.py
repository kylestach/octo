import os

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("bucket_path", None, "Path to save dir on GCP bucket")
flags.DEFINE_string("step_number", None, "Step number")
flags.DEFINE_string(
    "save_path", "/iliad2/u/jhejna/checkpoints/orca", "Path to save checkpoint"
)


def main(_):
    if FLAGS.bucket_path.startswith("gs"):
        cmd = "gsutil cp"
    else:
        cmd = "scp "
    checkpoint_path = os.path.join(FLAGS.bucket_path, f"{FLAGS.step_number}")
    norm_path = os.path.join(FLAGS.bucket_path, "dataset_statistics*")
    config_path = os.path.join(FLAGS.bucket_path, "config.json*")
    example_batch_path = os.path.join(FLAGS.bucket_path, "example_batch.msgpack*")
    run_name = os.path.basename(os.path.normpath(FLAGS.bucket_path))
    save_path = os.path.join(FLAGS.save_path, run_name)
    os.makedirs(save_path, exist_ok=True)

    os.system(f"{cmd} -r {checkpoint_path} {save_path}/")
    os.system(f"{cmd} {norm_path} {save_path}/")
    os.system(f"{cmd} {config_path} {save_path}/")
    os.system(f"{cmd} {example_batch_path} {save_path}/")


if __name__ == "__main__":
    app.run(main)
