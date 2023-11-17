from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.env_util import make_vec_env

from wagner_linear_environment import WagnerLinearEnvironment

if __name__ == '__main__':
    number_of_nodes=10
    env = WagnerLinearEnvironment(number_of_nodes)
    model = PPO("MlpPolicy", env, verbose=0)
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    model.save(f"WagnerLinearEnvironment{number_of_nodes}PPO{total_timesteps}")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            env.render()
            obs, info = env.reset()
            break