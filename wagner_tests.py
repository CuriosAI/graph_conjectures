from stable_baselines3 import PPO

from wagner_linear_environment import WagnerLinearEnvironment

if __name__ == '__main__':
    env = WagnerLinearEnvironment(18)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10_000_000, log_interval=4)
    model.save("WagnerLinearEnvironmentPPO10_000_000")
    env.render_mode = "human"
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            obs, info = env.reset()
            break