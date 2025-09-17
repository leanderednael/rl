import os
import pathlib
from typing import Callable, Optional, cast

import gymnasium as gym
import tqdm

from src.agents import AbstractAgent, Reward, State
from src.minihack_envs import get_env, hashable, plot_observations
from src.returns import calculate_episodic_return
from src.stable_baselines3_envs import PixelObservationWrapper


class RLTask:
    """This class abstracts the concept of an agent interacting with an environment.

    :param env: the environment to interact with (e.g. a gym.Env)
    :param agent: the interacting agent
    """

    def __init__(self, env: gym.Env, agent: AbstractAgent) -> None:
        self.env = env
        self.agent = agent

    def interact(self, num_episodes: int) -> list[float]:
        """This function executes num_episodes of interaction between the agent and the environment.

        :param num_episodes: the number of episodes of the interaction
        :return: a list of episode average returns
        """
        avg_returns: list[float] = []
        """.. math:: \hat{G}_k = \frac{1}{k+1}\sum_{i=0}^k{G_i}"""
        episodic_returns: list[float] = []

        for k in tqdm.tqdm(range(num_episodes)):
            self.env.reset()
            self.agent.reset_episode_history()

            t = 0
            reward: Reward = 0.0
            self.agent.R.append(reward)  # R[0] - just for indexing purposes
            done = False
            state = cast(State, self.env.reset())
            self.agent.S.append(hashable(state))  # S[0]
            action = self.agent.act(state)
            self.agent.A.append(action)  # A[0]

            while not done:
                observation, reward, terminated, truncated, info = self.env.step(action)  # type: ignore[assignment]
                self.agent.R.append(reward)  # R[t+1]
                done = terminated or truncated
                state = cast(State, (observation, info))
                self.agent.S.append(hashable(state))  # S[t+1]
                action = self.agent.act(state)
                self.agent.A.append(action)  # A[t+1]

                self.agent.on_step_end(t)
                t += 1

            episodic_returns.append(
                calculate_episodic_return(
                    self.agent.R[1:],  # R[1:] - rewards start at index 1
                    self.agent.gamma,
                )
            )
            avg_returns.append(sum(episodic_returns) / (k + 1))  # k is 0-indexed!
            self.agent.on_episode_end(k)

        return avg_returns

    def visualize_episode(
        self,
        max_num_steps: Optional[int] = None,
        *,
        custom_callback: Optional[Callable[[State, Optional[pathlib.Path]], None]] = None,
        save_fig: bool = False,
    ) -> None:
        """This function executes and plot an episode (or a fixed number 'max_num_steps' steps).

        :param max_num_steps: Optional, maximum number of steps to plot.
        :param custom_callback: Optional, a custom function to call at each step instead of rendering the environment.
        """
        self.env.reset()
        self.agent.reset_episode_history()
        self.agent.learning = False

        def fig_path(t: int) -> Optional[pathlib.Path]:
            if save_fig:
                return pathlib.Path(self.agent.id, str(t))
            return None

        t = 0
        done = False
        state = cast(State, self.env.reset())

        print(f"Step {t}:")
        custom_callback(state, fig_path(t)) if custom_callback else self.env.render()

        while not done and (max_num_steps is None or t < max_num_steps):
            t += 1
            action = self.agent.act(state)
            observation, _, terminated, truncated, info = self.env.step(action)
            state = cast(State, (observation, info))
            done = terminated or truncated

            print(f"Step {t}:")
            custom_callback(state, fig_path(t)) if custom_callback else self.env.render()


def visualize_episode(model, env_id, max_steps=100, save_fig=False):
    """
    Visualize an episode using a trained model, showing both the environment and observation pixels.

    Args:
        model: The trained RL model (DQN or PPO)
        env_id: Environment ID to create
        max_steps: Maximum number of steps to visualize
        save_fig: Whether to save the observation plots
    """
    # Create evaluation environment with wrapper to handle chars + pixels
    eval_env = PixelObservationWrapper(get_env(env_id, max_episode_steps=max_steps, add_pixels=True))

    # Run one episode
    obs, _ = eval_env.reset()
    done = False
    step = 0

    print(f"Starting visualization of {env_id}")

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        print(f"\nStep {step}: Action={action}, Reward={reward}")

        output_path = None
        if save_fig:
            os.makedirs(f"./tex/figures/{env_id}", exist_ok=True)
            output_path = pathlib.Path(f"./tex/figures/{env_id}/step_{step}.png")

        raw_obs = eval_env.get_raw_obs()
        plot_observations((raw_obs, info), save_fig=output_path)

        obs = next_obs
        step += 1

    print(f"Episode finished after {step} steps")
    eval_env.close()
