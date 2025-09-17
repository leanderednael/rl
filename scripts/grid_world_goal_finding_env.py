import matplotlib.pyplot as plt

from src import agents, grid_world_goal_finding_env, task


def grid_world_goal_finding_world(*, num_episodes: int, max_num_steps: int) -> None:
    grid_world_goal_finding_task = task.RLTask(
        grid_world_goal_finding_env.Env(5, 5),
        agents.RandomAgent(
            "grid-world-goal-finding_RandomAgent", action_space=grid_world_goal_finding_env.action_space
        ),
    )

    plt.figure()
    plt.plot(grid_world_goal_finding_task.interact(num_episodes))
    plt.title("Average Returns over Episodes: grid-world-goal-finding env, RandomAgent")
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    plt.show()
    grid_world_goal_finding_task.visualize_episode(max_num_steps)
