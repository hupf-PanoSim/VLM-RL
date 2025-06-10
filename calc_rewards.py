import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import config_vlm_rl as config

CONFIG = config.set_vlm_rl_config()
CONFIG.algorithm_params.device = 'cuda:0'

from clip.clip_rewarded_sac import CLIPRewardedSAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.envs.carla_route_env import CarlaRouteEnv
from PanoSim_env.envs.env import PanoSimEnv
from carla_env.state_commons import create_encode_state_fn
from utils import HParamCallback, TensorboardCallback, write_json
import numpy as np

min_speed = 20.0
max_speed = 35.0
target_speed = 25.0
max_distance = 3.0
max_std_center_lane = 0.4
max_angle_center_lane = 90
penalty_reward = -10
early_stop = True

def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor
    return reward

def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        # if early_stop:
        speed = env.vehicle.get_speed()
        if speed < 1.0:
            env.low_speed_timer += 1
        else:
            env.low_speed_timer = 0.0  # Reset timer if speed goes above threshold

        # Check if speed is low for 90 consecutive second
        if env.low_speed_timer >= 90 * env.fps:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Stop if distance from center > max distance
        if env.distance_from_center > max_distance and not env.eval:
            env.terminal_state = True
            terminal_reason = "Off-track"

        # Stop if speed is too high
        if max_speed > 0 and speed > max_speed and not env.eval:
            env.terminal_state = True
            terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            env.low_speed_timer = 0.0
            if reward_fn in {reward_fn5}:
                reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([terminal_reason, ""])
        return reward

    return func

reward_fn5 = create_reward_fn(reward_fn5)

log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)

use_carla = False
if use_carla:
    env = CarlaRouteEnv(
        obs_res                 =(80, 120),
        host                    ='localhost',
        port                    =2000,
        reward_fn               =reward_fn5,
        observation_space       =observation_space,
        encode_state_fn         =encode_state_fn,
        fps                     =15,
        action_smoothing        =0.75,
        action_space_type       ='continuous',
        activate_spectator      =False,
        activate_render         =False,
        activate_bev            =True,
        activate_seg_bev        =True,
        activate_traffic_flow   =True,
        start_carla             =True
    )
else:
    env = PanoSimEnv(
        observation_space       =observation_space,
        encode_state_fn         =encode_state_fn
    )

model = CLIPRewardedSAC(env=env, config=CONFIG)

model_suffix = "{}_id{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), 'vlm_rl')
model_name = f'{CLIPRewardedSAC.__name__}_{model_suffix}'
model_dir = os.path.join(log_dir, model_name)

new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

model.learn(
    total_timesteps     =1000000,
    callback            =[
        HParamCallback(CONFIG),
        TensorboardCallback(1),
        CheckpointCallback(
            save_freq   =10000,
            save_path   =model_dir,
            name_prefix ="model"
        )
    ],
    reset_num_timesteps =False
)


"""
print(CONFIG)
{
    'algorithm': 'CLIP-SAC',
    'algorithm_params': {
        'device': 'cuda:0',
        'learning_rate': 'lr_schedule(1e-4, 5e-7, 2)',
        'buffer_size': 100000,
        'batch_size': 256,
        'ent_coef': 'auto',
        'gamma': 0.98,
        'tau': 0.02,
        'train_freq': 64,
        'gradient_steps': 64,
        'learning_starts': 10000,
        'use_sde': True,
        'policy_kwargs': {
            'log_std_init': -3,
            'net_arch': [500, 300],
            'features_extractor_class': 'config_vlm_rl.CustomMultiInputExtractor',
            'features_extractor_kwargs': {
                'features_dim': 256
            }
        }
    },
    'state': ['steer', 'throttle', 'speed', 'waypoints', 'seg_camera'],
    'action_smoothing': 0.75,
    'reward_fn': 'reward_fn5',
    'reward_params': {
        'early_stop': True,
        'min_speed': 20.0,
        'max_speed': 35.0,
        'target_speed': 25.0,
        'max_distance': 3.0,
        'max_std_center_lane': 0.4,
        'max_angle_center_lane': 90,
        'penalty_reward': -10
    },
    'clip_reward_params': {
        'pretrained_model': 'ViT-bigG-14/laion2b_s39b_b160k',
        'batch_size': 64,
        'target_prompts': [
            'Two cars have collided with each other on the road',
            'The road is clear with no car accidents'
        ]
    },
    'vlm_reward_type': 'VLM-RL',
    'obs_res': (80, 120),
    'seed': 100,
    'wrappers': [],
    'action_noise': {},
    'use_seg_bev': True,
    'use_rgb_bev': True
}
"""