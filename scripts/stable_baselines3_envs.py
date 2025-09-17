from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src import minihack_envs, stable_baselines3_envs, task


def learning(*, num_episodes: int, max_num_steps: int) -> None:
    for env_id in [
        minihack_envs.EMPTY_ROOM,
        minihack_envs.ROOM_WITH_MULTIPLE_MONSTERS,
    ]:

        env = minihack_envs.get_env(env_id)
        env = stable_baselines3_envs.EncodingWrapper(env)
        env = Monitor(env)
        env.reset()

        for model in [
            PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                gamma=1,
                learning_rate=1e-4,
                ent_coef=0.01,
                tensorboard_log=f"./logs/{env_id}",
            ),
            DQN(
                "MultiInputPolicy",
                env,
                verbose=1,
                gamma=1,
                learning_rate=1e-4,
                buffer_size=100_000,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                train_freq=4,
                tensorboard_log=f"./logs/{env_id}",
            ),
        ]:
            model_path = f"./models/{env_id}/{model.__class__.__name__}"
            print(model.policy)
            model.learn(
                total_timesteps=100_000,
                log_interval=1,
                progress_bar=True,
                tb_log_name="PPO",
            )

            model.save(model_path)
            print(f"Model saved to {model_path}")

            # model = PPO.load(model_path)
            # print(f"Model loaded from {model_path}")
            env.reset()

            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
            print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

            task.visualize_episode(model, env_id, max_steps=max_num_steps)
