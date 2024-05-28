import rail_walker_gym.envs.register_mujoco
from training.task_config_util import apply_task_configs
from training.task_configs.default import get_config as get_task_config
from training.configs.reset_config import get_config as get_reset_config

import gym

class Go1Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def get_instruction(self):
        return "walk"

    def process_obs(self, proprio):
        obs = {"proprio_quadruped": proprio}
        # return image just for rendering
        obs["image_video"] = self.env.render()
        return obs

    def step(self, action):
        self.ep_len += 1
        proprio, reward, done, trunc, info = self.env.step(action)
        obs = self.process_obs(proprio)
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.ep_len = 0
        proprio, info = self.env.reset(return_info=True, **kwargs)
        obs = self.process_obs(proprio)
        return obs, info

    def get_episode_metrics(self):
        return {"episode_length": self.ep_len}

def make_go1_env(**kwargs):
    name = "Go1SanityMujoco-Empty-SepRew-v0"

    task_config = get_task_config()
    task_config["action_interpolation"] = True
    task_config["enable_reset_policy"] = False
    task_config["Kp"] = 20
    task_config["Kd"] = 1.0
    task_config["limit_episode_length"] = 300
    task_config["action_range"] = 0.35
    task_config["frame_stack"] = 0
    task_config["action_history"] = 1
    task_config["rew_target_velocity"] = 1.5
    task_config["rew_energy_penalty_weight"] = 0.0
    task_config["rew_qpos_penalty_weight"] = 2.0
    task_config["rew_smooth_torque_penalty_weight"] = 0.005
    task_config["rew_pitch_rate_penalty_factor"] = 0.4
    task_config["rew_roll_rate_penalty_factor"] = 0.2
    task_config["rew_joint_diagonal_penalty_weight"] = 0.00
    task_config["rew_joint_shoulder_penalty_weight"] = 0.00
    task_config["rew_joint_acc_penalty_weight"] = 0.0
    task_config["rew_joint_vel_penalty_weight"] = 0.0
    task_config["center_init_action"] = True
    task_config["rew_contact_reward_weight"] = 0.0
    task_config["filter_actions"] = 8
    task_config["rew_smooth_change_in_tdy_steps"] = 1

    reset_agent_config = get_reset_config()

    env = gym.make(name)
    _, env = apply_task_configs(env, name, 0, task_config, reset_agent_config, True)
    return Go1Wrapper(env)

# register gym environment
gym.register(
    "go1",
    entry_point=make_go1_env
)
