from unittest import mock

import gymnasium as gym
import pytest

from src.agents import AbstractEpsilonGreedyLearningAgent


@pytest.fixture
def action_space() -> gym.spaces.Discrete:
    space = mock.MagicMock(spec=gym.spaces.Discrete)
    space.n = 4
    return space


def test_epsilon_scheduling_initialization(action_space: gym.spaces.Discrete) -> None:
    epsilon_start = 1.0
    epsilon_end = 0.1
    max_num_episodes = 100

    agent = AbstractEpsilonGreedyLearningAgent(
        id="test_agent",
        action_space=action_space,
        epsilon=epsilon_start,
        epsilon_scheduling=epsilon_end,
        max_num_episodes=max_num_episodes,
    )

    assert agent.epsilon == epsilon_start
    assert agent.eps_sched is not None
    assert agent.eps_sched.epsilon_start == epsilon_start
    assert agent.eps_sched.epsilon_end == epsilon_end
    assert agent.eps_sched.max_num_episodes == max_num_episodes


def test_epsilon_scheduling_decay(action_space: gym.spaces.Discrete) -> None:
    epsilon_start = 1.0
    epsilon_end = 0.1
    max_num_episodes = 100

    agent = AbstractEpsilonGreedyLearningAgent(
        id="test_agent",
        action_space=action_space,
        epsilon=epsilon_start,
        epsilon_scheduling=epsilon_end,
        max_num_episodes=max_num_episodes,
    )

    def expected_epsilon(k: int) -> float:
        return epsilon_start - k * ((epsilon_start - epsilon_end) / max_num_episodes)

    assert agent.epsilon == epsilon_start

    for episode in [0, 1, 2, 3, 4, 5, 25, 50, 75, 99, 100, 101]:
        agent.on_episode_end(episode)

        expected = max(epsilon_end, min(epsilon_start, expected_epsilon(episode)))
        assert pytest.approx(agent.epsilon) == expected, f"Epsilon value incorrect for episode {episode}"


def test_epsilon_scheduling_no_decay(action_space: gym.spaces.Discrete) -> None:
    epsilon_start = 0.5
    epsilon_end = 0.5
    max_num_episodes = 100

    agent = AbstractEpsilonGreedyLearningAgent(
        id="test_agent",
        action_space=action_space,
        epsilon=epsilon_start,
        epsilon_scheduling=epsilon_end,
        max_num_episodes=max_num_episodes,
    )

    for episode in [0, 50, 100]:
        agent.on_episode_end(episode)
        assert agent.epsilon == epsilon_start


def test_epsilon_scheduling_immediate_decay(action_space: gym.spaces.Discrete) -> None:
    epsilon_start = 1.0
    epsilon_end = 0.1
    max_num_episodes = 1

    agent = AbstractEpsilonGreedyLearningAgent(
        id="test_agent",
        action_space=action_space,
        epsilon=epsilon_start,
        epsilon_scheduling=epsilon_end,
        max_num_episodes=max_num_episodes,
    )

    assert agent.epsilon == epsilon_start

    agent.on_episode_end(0)
    assert agent.epsilon == epsilon_start

    agent.on_episode_end(1)
    assert pytest.approx(agent.epsilon) == epsilon_end


def test_epsilon_scheduling_requires_max_num_episodes(action_space: gym.spaces.Discrete) -> None:
    with pytest.raises(ValueError, match="'max_num_episodes' must be provided for 'epsilon_scheduling'"):
        AbstractEpsilonGreedyLearningAgent(
            id="test_agent", action_space=action_space, epsilon=0.5, epsilon_scheduling=0.1, max_num_episodes=None
        )


def test_epsilon_scheduling_with_learning_disabled(action_space: gym.spaces.Discrete) -> None:
    epsilon_start = 1.0
    epsilon_end = 0.1
    max_num_episodes = 100

    agent = AbstractEpsilonGreedyLearningAgent(
        id="test_agent",
        action_space=action_space,
        epsilon=epsilon_start,
        epsilon_scheduling=epsilon_end,
        max_num_episodes=max_num_episodes,
    )

    agent.learning = False

    initial_epsilon = agent.epsilon
    for episode in [0, 25, 50, 75, 99, 100, 101]:
        agent.on_episode_end(episode)
        assert agent.epsilon == initial_epsilon
