"""TODO: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import FlattenObservation
from minihack.envs.room import MiniHackRoom
from nle import nethack
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

if torch.cuda.is_available():
    torch.device("cuda")
elif torch.mps.is_available():
    torch.device("mps")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ExplorationRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.visited = set()

    def reset(self, **kwargs):
        self.visited.clear()
        obs, info = self.env.reset(**kwargs)
        self.visited.add(self._pos(info))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = self._pos(info)
        if pos not in self.visited:
            reward += 1.0  # bonus for exploring new tile
            self.visited.add(pos)
        return obs, reward, terminated, truncated, info

    def _pos(self, info):
        # Use cursor position as proxy for agent position
        return tuple(info.get("blstats", [])[0:2])


class CustomCNN(BaseFeaturesExtractor):
    """

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CharObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space["chars"].shape
        # Add a channel dimension for CNN input (C, H, W)
        self.observation_space = Box(low=0, high=255, shape=(1, *obs_shape), dtype=np.uint8)

    def observation(self, obs):
        chars = obs["chars"].astype(np.uint8)
        return chars[None, :, :]  # Add channel dimension (1, H, W)


def create_env():
    env = MiniHackRoom(
        size=5,
        max_episode_steps=50,
        actions=tuple(nethack.CompassCardinalDirection),
        random=False,
        observation_keys=["chars"],
        reward_win=0,
        penalty_step=-1,
        penalty_time=-1,
    )
    return env


mlp_env = DummyVecEnv([lambda: FlattenObservation(create_env())])
mlp_policy = "MlpPolicy"
"""The MlpPolicy treats the input observation as a flat vector, losing the 2D spatial information inherent in the chars observation grid.
This makes it difficult for the agent to understand concepts like corners or relative positions within the room."""

cnn_env = CharObsWrapper(create_env())
cnn_policy = "CnnPolicy"

if True:
    env = cnn_env
    policy = cnn_policy
else:
    env = mlp_env
    policy = mlp_policy

env = ExplorationRewardWrapper(env)  # type: ignore[assignment]

check_env(env)

# model = DQN("MlpPolicy", mlp_env, buffer_size=350_000)
model = DQN(
    cnn_policy,
    env,
    policy_kwargs=dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128)),
    # buffer_size=350_000,
    seed=SEED,
    # target_update_interval=100,
)
model.learn(total_timesteps=1000, progress_bar=True)

obs, infos = env.reset()
env.render()
for _ in range(10):
    actions, _states = model.predict(obs, deterministic=True)
    obs, _rewards, terminated, truncated, infos = env.step(actions)
    env.render()
    if terminated or truncated:
        break
env.close()
