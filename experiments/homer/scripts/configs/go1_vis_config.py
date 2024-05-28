from ml_collections import ConfigDict



def get_config():

    base_rollout_vis_kwargs = dict(
        env_name="go1",
        max_episode_length=300,
        exec_horizon=1,
        history_length=5,
        vis_fps=10,
        video_subsample_rate=5,
        use_temp_ensembling=False,
        env_kwargs=dict(),
        video_obs_key="image_video",
        head_name="quadruped"
    )

    config = ConfigDict(
        dict(
            rollout_kwargs=dict(
                dataset_name="go1",
                modes_to_evaluate=("text_conditioned",),
                trajs_for_rollouts=10,
                visualizer_kwargs_list=[
                    dict(
                        **base_rollout_vis_kwargs,
                        name="go1",
                    )
                ]
            )
        )
    )

    return config
