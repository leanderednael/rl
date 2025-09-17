import gymnasium as gym
import numpy as np

from src import minihack_envs

encoding = {
    minihack_envs.AGENT: 1,
    minihack_envs.FREE: 0,
    minihack_envs.GOAL: 2,
    minihack_envs.START: 3,
}


def encode_observation(observation):
    """Encodes the observation using the encoding dictionary."""
    chars = observation["chars"]
    encoded_chars = np.array([[encoding.get(val, 0) for val in row] for row in chars])
    observation["chars"] = encoded_chars  # Update the chars in the observation with the encoded values
    return observation


class EncodingWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return encode_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return encode_observation(obs), reward, terminated, truncated, info


class PixelObservationWrapper(gym.Wrapper):
    """
    Wrapper to handle environments with both chars and pixels observations,
    but only pass chars to the model for prediction.
    """

    def __init__(self, env):
        super().__init__(env)
        self._raw_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._raw_obs = obs  # Store the full observation

        if isinstance(obs, dict) and "chars" in obs:
            filtered_obs = {"chars": obs["chars"]}
            return filtered_obs, info
        return encode_observation(obs), info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._raw_obs = next_obs  # Store the full observation
        # Return only the chars part for the model
        if isinstance(next_obs, dict) and "chars" in next_obs:
            filtered_obs = {"chars": next_obs["chars"]}
            return filtered_obs, reward, terminated, truncated, info
        return encode_observation(next_obs), reward, terminated, truncated, info

    def get_raw_obs(self):
        """Return the raw observation with pixels for visualization"""
        return self._raw_obs
