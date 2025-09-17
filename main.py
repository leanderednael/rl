from scripts.grid_world_goal_finding_env import grid_world_goal_finding_world
from scripts.minihack_envs import EnvironmentConfig, compare, minihack_worlds
from scripts.stable_baselines3_envs import learning
from src import agents, minihack_envs

if __name__ == "__main__":
    match "compare_dyna_q":
        case "grid_world_goal_finding_world":
            grid_world_goal_finding_world(num_episodes=10_000, max_num_steps=10)
        case "minihack_worlds":
            minihack_worlds(max_num_steps=10, save_fig=True)
        case "compare_mc_td":
            compare(
                [
                    agents.MCAgent,
                    agents.SARSAAgent,
                ],
                [
                    EnvironmentConfig(minihack_envs.EMPTY_ROOM, 2500, 15, size=5),
                    EnvironmentConfig(minihack_envs.EMPTY_ROOM, 2500, 15, size=7),
                    EnvironmentConfig(minihack_envs.CLIFF, 2500, 10),
                ],
            )
        case "compare_every_visit":
            compare(
                [agents.MCAgent],
                [EnvironmentConfig(minihack_envs.EMPTY_ROOM, 2500, 15, size=7)],
                every_visit=True,
            )
        case "comapre_on_off_policy":
            compare(
                [
                    agents.SARSAAgent,
                    agents.QAgent,
                ],
                [
                    EnvironmentConfig(minihack_envs.EMPTY_ROOM, 1000, 10),
                    EnvironmentConfig(minihack_envs.CLIFF, 1000, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_LAVA, 1000, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MONSTER, 1000, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MULTIPLE_MONSTERS, 1000, 10),
                ],
            )
        case "compare_exploration_rates":
            compare(
                [
                    agents.SARSAAgent,
                    agents.QAgent,
                    agents.DynaQAgent,
                ],
                [
                    EnvironmentConfig(minihack_envs.CLIFF, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_LAVA, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MONSTER, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MULTIPLE_MONSTERS, 700, 10),
                ],
                exploration_rate=[0.1, 0.4, 0.8],
            )
        case "compare_learning_rates":
            compare(
                [
                    agents.MCAgent,
                    agents.SARSAAgent,
                    agents.QAgent,
                    agents.DynaQAgent,
                ],
                [
                    EnvironmentConfig(minihack_envs.CLIFF, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_LAVA, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MONSTER, 700, 10),
                    # EnvironmentConfig(minihack_envs.ROOM_WITH_MULTIPLE_MONSTERS, 700, 10),
                ],
                learning_rate=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            )
        case "compare_dyna_q":
            compare(
                [
                    agents.QAgent,
                    agents.DynaQAgent,
                ],
                [
                    EnvironmentConfig(minihack_envs.EMPTY_ROOM, 1000, 10),
                    EnvironmentConfig(minihack_envs.CLIFF, 1000, 10),
                ],
            )
        case "compare_dqn_ac":
            learning(num_episodes=700, max_num_steps=10)
