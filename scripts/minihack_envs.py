from typing import Iterable, NamedTuple, Optional, Sequence, cast

import matplotlib.pyplot as plt

from src import agents, minihack_envs, task


def minihack_worlds(*, max_num_steps: int, save_fig: bool = False) -> None:
    for env_id in [minihack_envs.EMPTY_ROOM, minihack_envs.ROOM_WITH_LAVA]:
        task.RLTask(
            minihack_envs.get_env(id=env_id, add_pixels=True),
            agents.FixedAgent(f"{env_id}_FixedAgent", action_space=minihack_envs.action_space),
        ).visualize_episode(max_num_steps, custom_callback=minihack_envs.plot_observations, save_fig=save_fig)


class EnvironmentConfig(NamedTuple):
    env_id: str
    num_episodes: int
    max_num_steps: int
    size: int = 5


def compare(
    Agents: Sequence[type[agents.AbstractAgent]],
    environments: Sequence[EnvironmentConfig],
    *,
    evaluate: bool = True,
    exploration_rate: Optional[Iterable[float]] = None,
    learning_rate: Optional[Iterable[float]] = None,
    every_visit: Optional[bool] = False,
) -> None:
    for env_id, num_episodes, max_num_steps, size in environments:
        for epsilon in exploration_rate or [0.05]:
            for alpha in learning_rate or [0.1]:
                agents_avg_returns = {}
                if every_visit:
                    Agents = list(Agents) * 2
                for idx, Agent in enumerate(cast(list[type[agents.AbstractAgent]], Agents)):
                    if every_visit and idx % 2 == 1:
                        agent_name = str(Agent).replace("Agent", "EveryVisitAgent")
                    else:
                        agent_name = str(Agent)
                    agent = Agent(
                        f"{env_id}_{agent_name}",
                        action_space=minihack_envs.action_space,
                        every_visit=True if idx % 2 == 1 else False,
                        epsilon=epsilon,
                        alpha=alpha,
                    )

                    learning_env = minihack_envs.get_env(env_id, add_pixels=False, size=size)
                    avg_returns = task.RLTask(learning_env, agent).interact(num_episodes)

                    agents_avg_returns[agent_name] = avg_returns

                    if evaluate:
                        eval_env = minihack_envs.get_env(env_id, add_pixels=True, size=size)
                        task.RLTask(eval_env, agent).visualize_episode(
                            max_num_steps, custom_callback=minihack_envs.plot_observations
                        )

                plt.figure()
                for agent_name, avg_returns in agents_avg_returns.items():
                    plt.plot(avg_returns, label=agent_name)
                plt.legend()
                if exploration_rate and learning_rate:
                    title_suffix = f"(epsilon: {epsilon}, alpha: {alpha})"
                elif exploration_rate:
                    title_suffix = f"(epsilon: {epsilon})"
                elif learning_rate:
                    title_suffix = f"(alpha: {alpha})"
                else:
                    title_suffix = ""
                plt.title(f"Average Returns over Episodes: {env_id} env {title_suffix}")
                plt.xlabel("Episodes")
                plt.ylabel("Average Return")
                plt.grid(True, linestyle="--")
                plt.show()
