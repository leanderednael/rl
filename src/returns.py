from typing import Sequence, SupportsFloat

import numpy as np


def calculate_returns(rewards: Sequence[SupportsFloat], gamma: float = 1.0) -> list[float]:
    """Calculate discounted returns for each step.

    .. math::
        G_t = \sum_{l=0}^\infty{\gamma^l R_{t+1+l}}

    :param rewards: sequence of rewards.
    :param gamma: discount factor (0 <= gamma <= 1).
    :return: sequence of discounted returns for each time step.
    """
    if not rewards:
        return []

    T = len(rewards)

    discount_factors = np.power(gamma, np.arange(T))  # [1, gamma, gamma^2, ...]
    discount_matrix = np.zeros((T, T))
    for i in range(T):
        discount_matrix[i, i:] = discount_factors[: T - i]

    returns = discount_matrix @ np.array(rewards)

    return returns.tolist()


def calculate_episodic_return(rewards: Sequence[SupportsFloat], gamma: float = 1.0) -> float:
    """Calculate the discounted return for the entire episode.

    .. math::
        G = \sum_{t=0}^{T-1}{\gamma^t R_{t+1}}

    :param rewards: sequence of rewards.
    :param gamma: discount factor (0 <= gamma <= 1).
    :return: the discounted return value for the entire episode.
    """
    if not rewards:
        return 0.0

    T = len(rewards)
    discount_factors = np.power(gamma, np.arange(T))  # [1, gamma, gamma^2, ...]

    episode_return = np.sum(discount_factors * np.array(rewards))

    return float(episode_return)
