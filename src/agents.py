import random
from collections import defaultdict
from typing import NamedTuple, Optional

import gymnasium as gym
import numpy as np

from src.minihack_envs import ActIndex, HashableState, State, hashable


class EpsilonScheduling(NamedTuple):
    """Epsilon scheduling for epsilon-greedy action selection.

    :param epsilon_start: initial value for epsilon-greedy action selection
    :param epsilon_end: final value for epsilon-greedy action selection
    :param max_num_episodes: the number of episodes to reach the final value
    """

    epsilon_start: float
    epsilon_end: float
    max_num_episodes: int


Reward = float
QValue = tuple[HashableState, ActIndex]
"""Expected return from taking action a in state s, then following the (sub/optimal; on/off-policy learning) policy thereafter"""


class AbstractAgent:
    """An abstract interface for an agent.

    :param id: a str-unique identifier for the agent
    :param action_space: the actions that an agent can take
    :param alpha: the learning rate
    :param gamma: the discount factor
    :param epsilon: initial value for epsilon-greedy action selection
    :param epsilon_scheduling: final value for epsilon-greedy action selection, or None if no epsilon scheduling
    :param max_num_episodes: must be provided if using epsilon scheduling
    :param every_visit: consider every visit (or only first visit) in MC methods
    :param num_planning_steps: the number of planning steps for model-based planning agents
    """

    def __init__(
        self,
        id: str,
        action_space: gym.spaces.Discrete,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.05,
        epsilon_scheduling: Optional[float] = None,
        max_num_episodes: Optional[int] = None,
        every_visit: float = False,
        num_planning_steps: int = 10,
    ) -> None:
        self.id = id
        self.action_space = action_space

        self.learning: bool = True
        """Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        You may want to disable some behaviour when not learning (e.g. no update rule, no exploration epsilon = 0, etc.)"""
        self.reset_episode_history()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_sched: Optional[EpsilonScheduling] = None
        if epsilon_scheduling:
            if not max_num_episodes:
                raise ValueError("'max_num_episodes' must be provided for 'epsilon_scheduling'")

            self.eps_sched = EpsilonScheduling(
                epsilon_start=epsilon,
                epsilon_end=epsilon_scheduling,
                max_num_episodes=max_num_episodes,
            )

        self.Q: defaultdict[QValue, float] = defaultdict(float)
        """State-action quality values, Q-Table (across episodes - not reset at end of episode!)"""
        self.cross_episode_state_action_count: defaultdict[QValue, int] = defaultdict(int)
        """State-action count table (across episodes - not reset at end of episode!)"""

        self.every_visit = every_visit
        self.num_planning_steps = num_planning_steps

    def reset_episode_history(self) -> None:
        """Temporary buffer holding information for the current episode."""
        self.S: list[HashableState] = []
        self.A: list[ActIndex] = []
        self.R: list[Reward] = []
        self.model: dict[QValue, tuple[Reward, HashableState]] = {}

    def __str__(self) -> str:
        if isinstance(self, type):
            return self.__name__
        return self.__class__.__name__

    def act(self, state: State) -> ActIndex:
        """Policy for action selection.

        :param state: the state on which to act

        :return action: the action to take in the given state
        """
        raise NotImplementedError

    def on_step_end(self, t: int) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related)
        at the end of a step.

        :param t: the time step
        """
        pass

    def on_episode_end(self, k: int) -> None:
        """This function can be exploited to allow the agent to perform some internal process (e.g. learning-related)
        at the end of an episode.

        :param k: the episode
        """
        pass


class RandomAgent(AbstractAgent):
    """Random Policy"""

    def act(self, state: State) -> ActIndex:
        return int(self.action_space.sample())


class FixedAgent(AbstractAgent):
    """Hardcoded Policy"""

    def act(self, state: State) -> ActIndex:
        from src import minihack_envs

        observation, _ = state
        chars = minihack_envs.get_crop_chars_from_observation(observation)

        agent_position = np.where(chars == minihack_envs.AGENT)
        if agent_position[0].size == 0 or agent_position[1].size == 0:
            raise ValueError(f"Agent not found in the observation: {chars}.")
        agent_row, agent_col = agent_position[0][0], agent_position[1][0]

        if not (agent_row == chars.shape[0] - 1 or chars[agent_row + 1, agent_col] != minihack_envs.FREE):
            return minihack_envs.ACTIONS.index(minihack_envs.ACTION.S)
        return minihack_envs.ACTIONS.index(minihack_envs.ACTION.E)


class AbstractEpsilonGreedyLearningAgent(AbstractAgent):
    def act(self, state: State) -> ActIndex:
        """Epsilon-greedy policy for action selection.

        :param state: the state on which to act

        :return action: the action to take in the given state
        """
        A = range(self.action_space.n)

        if not self.learning or random.random() > self.epsilon:
            S_t = hashable(state)
            Q = [self.Q[(S_t, a)] for a in A]
            max_action_value = max(Q)
            action = random.choice([a for a, q in enumerate(Q) if q == max_action_value])
        else:
            action = random.choice(list(A))

        return action

    def on_episode_end(self, k: int) -> None:
        if self.learning and self.eps_sched:
            decay_step = min(k, self.eps_sched.max_num_episodes)
            self.epsilon = self.eps_sched.epsilon_start - decay_step * (
                (self.eps_sched.epsilon_start - self.eps_sched.epsilon_end) / self.eps_sched.max_num_episodes
            )
        self.reset_episode_history()


class MCAgent(AbstractEpsilonGreedyLearningAgent):
    """Monte Carlo On-Policy

    Model-free, incremental implementation for sample averages
    - O(1) time complexity (vs. O(n) if storing returns G_t)
    - but less flexible for weighting of returns, e.g. recency-weighted averaging"""

    def on_episode_end(self, k: int) -> None:
        if not (len(self.S) == len(self.A) == len(self.R)):
            raise ValueError(
                f"The lengths of S ({len(self.S)}), A ({len(self.A)}), and R ({len(self.R)}) must be equal at the end of an episode."
            )
        T = len(self.S) - 1
        G_t = self.R[T]

        for t in range(T - 1, -1, -1):
            S_t, A_t = self.S[t], self.A[t]
            G_t = self.R[t + 1] + self.gamma * G_t

            if self.every_visit is True or not any(
                self.S[t_prev] == S_t and self.A[t_prev] == A_t for t_prev in range(0, t)
            ):
                self.cross_episode_state_action_count[(S_t, A_t)] += 1
                n = self.cross_episode_state_action_count[(S_t, A_t)]
                self.Q[(S_t, A_t)] = self.Q[(S_t, A_t)] + (1 / n) * (G_t - self.Q[(S_t, A_t)])

        super().on_episode_end(k)


class SARSAAgent(AbstractEpsilonGreedyLearningAgent):
    """Temporal Difference On-Policy (SARSA)"""

    def on_step_end(self, t: int) -> None:
        self.Q[(self.S[t], self.A[t])] += self.alpha * (
            self.R[t + 1] + self.gamma * self.Q[(self.S[t + 1], self.A[t + 1])] - self.Q[(self.S[t], self.A[t])]
        )


class QAgent(AbstractEpsilonGreedyLearningAgent):
    """Temporal Difference Off-Policy (Q-learning)"""

    def on_step_end(self, t: int) -> None:
        A = range(self.action_space.n)

        self.Q[(self.S[t], self.A[t])] += self.alpha * (
            self.R[t + 1] + self.gamma * max([self.Q[(self.S[t + 1], a)] for a in A]) - self.Q[(self.S[t], self.A[t])]
        )


class DynaQAgent(AbstractEpsilonGreedyLearningAgent):
    """Dyna-Q with Background Model-Based Planning Strategy"""

    def on_step_end(self, t: int) -> None:
        S_t, A_t = self.S[t], self.A[t]  # acting
        R_t_plus_1, S_t_plus_1 = self.R[t + 1], self.S[t + 1]  # experience
        A = range(self.action_space.n)

        self.Q[(S_t, A_t)] += self.alpha * (
            R_t_plus_1 + self.gamma * max([self.Q[(S_t_plus_1, a)] for a in A]) - self.Q[(S_t, A_t)]
        )  # direct RL
        self.model[(S_t, A_t)] = (R_t_plus_1, S_t_plus_1)  # model learning

        for _ in range(self.num_planning_steps):  # planning
            S_t, A_t = random.choice(list(self.model.keys()))
            R_t_plus_1, S_t_plus_1 = self.model[(S_t, A_t)]
            A = range(self.action_space.n)

            self.Q[(S_t, A_t)] += self.alpha * (
                R_t_plus_1 + self.gamma * max([self.Q[(S_t_plus_1, a)] for a in A]) - self.Q[(S_t, A_t)]
            )
