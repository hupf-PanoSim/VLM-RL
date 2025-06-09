from box import Box
from utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch

class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]

        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 58, 38)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 28, 18)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 13, 8)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 6, 4)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 2)
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).view(-1).shape[0]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x

class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key == "seg_camera":
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))
        return torch.cat(encoded_tensor_list, dim=1)


SAC_CLIP = dict(
    device="cuda:0",
    learning_rate=lr_schedule(1e-4, 5e-7, 2),
    buffer_size=100000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.98,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    learning_starts=10000,
    use_sde=True,
    policy_kwargs=dict(
        log_std_init=-3, net_arch=[500, 300],
        features_extractor_class=CustomMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
)

reward_fn_5_default = dict(
    early_stop=True,
    min_speed=20.0,  # km/h
    max_speed=35.0,  # km/h
    target_speed=25.0,  # kmh
    max_distance=3.0,  # Max distance from center before terminating
    max_std_center_lane=0.4,
    max_angle_center_lane=90,
    penalty_reward=-10,
)

reward_clg = dict(
    pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
    batch_size=64,
    target_prompts=[
        "Two cars have collided with each other on the road",
        "The road is clear with no car accidents",
    ],
)

CONFIG_vlm_rl = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": SAC_CLIP,
    "state": ["steer", "throttle", "speed", "waypoints", "seg_camera"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_fn_5_default,
    "clip_reward_params": reward_clg,
    "vlm_reward_type": "VLM-RL",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

CONFIG = None

def set_vlm_rl_config():
    global CONFIG
    CONFIG = Box(CONFIG_vlm_rl, default_box=True)
    return CONFIG
