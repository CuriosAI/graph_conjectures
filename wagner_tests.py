from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.env_util import make_vec_env

from wagner_linear_environment import WagnerLinearEnvironment

if __name__ == '__main__':
    number_of_nodes=18
    env = WagnerLinearEnvironment(number_of_nodes,dense_reward=False, penalize_star=True)
    total_timesteps = 20_000
    
    # model = PPO("MlpPolicy", env, verbose=0)
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=dict(net_arch=[2*env.number_of_edges,2*env.number_of_edges,2*env.number_of_edges, 2*env.number_of_edges,64,32]))
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    env.render_best_graph()
    model.save(f"WagnerLinearEnvironment{number_of_nodes}PPO{total_timesteps}")
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(reward)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         break