import numpy as np

from src.returns import calculate_episodic_return, calculate_returns


class TestCalculateDiscountedReturns:
    def test_gamma(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 0.9
        expected = [
            1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0,
            2.0 + 0.9 * 3.0 + 0.9**2 * 4.0,
            3.0 + 0.9 * 4.0,
            4.0,
        ]
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_almost_equal(returns, expected)

    def test_empty_rewards(self):
        rewards = []
        gamma = 0.9
        returns = calculate_returns(rewards, gamma)
        assert len(returns) == 0

    def test_single_reward(self):
        rewards = [5.0]
        gamma = 0.9
        returns = calculate_returns(rewards, gamma)
        assert returns[0] == 5.0

    def test_gamma_zero(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 0
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_array_equal(returns, rewards)

    def test_gamma_one(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 1.0
        expected = [
            1.0 + 2.0 + 3.0 + 4.0,
            2.0 + 3.0 + 4.0,
            3.0 + 4.0,
            4.0,
        ]
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_array_equal(returns, expected)

    def test_gamma_default(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        expected = [
            1.0 + 2.0 + 3.0 + 4.0,
            2.0 + 3.0 + 4.0,
            3.0 + 4.0,
            4.0,
        ]
        returns = calculate_returns(rewards)
        np.testing.assert_array_equal(returns, expected)

    def test_negative_rewards(self):
        rewards = [-1.0, -2.0, -3.0, -4.0]
        gamma = 0.9
        expected = [
            -1.0 - 0.9 * 2.0 - 0.9**2 * 3.0 - 0.9**3 * 4.0,
            -2.0 - 0.9 * 3.0 - 0.9**2 * 4.0,
            -3.0 - 0.9 * 4.0,
            -4.0,
        ]
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_almost_equal(returns, expected)

    def test_mixed_rewards(self):
        rewards = [1.0, -2.0, 3.0, -4.0]
        gamma = 0.9
        expected = [
            1.0 + 0.9 * -2.0 + 0.9**2 * 3.0 + 0.9**3 * (-4.0),
            -2.0 + 0.9 * 3.0 + 0.9**2 * (-4.0),
            3.0 + 0.9 * (-4.0),
            -4.0,
        ]
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_almost_equal(returns, expected)

    def test_different_gamma(self):
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.5
        expected = [
            1.0 + 0.5 * 2.0 + 0.5**2 * 3.0,
            2.0 + 0.5 * 3.0,
            3.0,
        ]
        returns = calculate_returns(rewards, gamma)
        np.testing.assert_almost_equal(returns, expected)


class TestCalculateDiscountedEpisodeReturn:
    def test_gamma(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 0.9
        expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
        episode_return = calculate_episodic_return(rewards, gamma)
        np.testing.assert_almost_equal(episode_return, expected)

    def test_empty_rewards(self):
        rewards = []
        gamma = 0.9
        episode_return = calculate_episodic_return(rewards, gamma)
        assert episode_return == 0.0

    def test_single_reward(self):
        rewards = [5.0]
        gamma = 0.9
        episode_return = calculate_episodic_return(rewards, gamma)
        assert episode_return == 5.0

    def test_gamma_zero(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 0
        episode_return = calculate_episodic_return(rewards, gamma)
        assert episode_return == rewards[0]

    def test_gamma_one(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        gamma = 1.0
        expected = sum(rewards)
        episode_return = calculate_episodic_return(rewards, gamma)
        assert episode_return == expected

    def test_gamma_default(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        expected = sum(rewards)
        episode_return = calculate_episodic_return(rewards)
        assert episode_return == expected

    def test_negative_rewards(self):
        rewards = [-1.0, -2.0, -3.0, -4.0]
        gamma = 0.9
        expected = -1.0 - 0.9 * 2.0 - 0.9**2 * 3.0 - 0.9**3 * 4.0
        episode_return = calculate_episodic_return(rewards, gamma)
        np.testing.assert_almost_equal(episode_return, expected)

    def test_mixed_rewards(self):
        rewards = [1.0, -2.0, 3.0, -4.0]
        gamma = 0.9
        expected = 1.0 + 0.9 * -2.0 + 0.9**2 * 3.0 + 0.9**3 * (-4.0)
        episode_return = calculate_episodic_return(rewards, gamma)
        np.testing.assert_almost_equal(episode_return, expected)

    def test_different_gamma(self):
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.5
        expected = 1.0 + 0.5 * 2.0 + 0.5**2 * 3.0
        episode_return = calculate_episodic_return(rewards, gamma)
        np.testing.assert_almost_equal(episode_return, expected)
