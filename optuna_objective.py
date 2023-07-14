from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import json

def objective(trial):
    # Define a Gym environment
    env = DummyVecEnv([lambda: gym.make('LinEnvMau-v0')])  # Replace with your environment

    # Define the hyperparameters to optimize, they can be taken looking at the signature of PPO at https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html. The following values are taken from https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe.
    # See the detailed explanation of hyperparameters at https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
    learning_rate = trial.suggest_float('learning_rate', 5e-6, 0.003, log=False)
    # Value of n_steps are multiple of 16, it is needed for next batch_size. Update: does not work, thus n_steps can be whatever.
    n_steps = trial.suggest_int('n_steps', 32, 5000, step=16)

    # The following does not work, because suggest_categorical does not support dynamically changes, see https://github.com/optuna/optuna/issues/372#issuecomment-480690635
    # # Create a list of numbers that are powers of 2 and divide n_steps
    # batch_sizes = list([i for i in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if n_steps % i == 0])
    # # Suggest a batch size from the list
    # batch_size = trial.suggest_categorical('batch_size', batch_sizes)

    # This workaround might bias the optimization process towards smaller batch sizes, because larger batch sizes are more likely to be adjusted.
    # Suggest a batch size as a power of 2
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    # Update: this workaround does not work, because after suggest has been performed, Optuna will use that value, and the next changes are useful. Thus, commenting the following code.
    # Adjust the batch size if it doesn't divide n_steps
    # while n_steps % batch_size != 0:
    #     # Decrease the batch size to the nearest smaller power of 2
    #     batch_size = batch_size // 2

    #n_steps = trial.suggest_int('n_steps', 16, 4096)
    #batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512, 1024])
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    gamma = trial.suggest_float('gamma', 0.8, 0.9997)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3])
    normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False]) # My suggestion
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)
    vf_coef = trial.suggest_categorical('vf_coef', [0.5, 1])
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.35, 0.7) # My suggestion
    target_kl = trial.suggest_float('target_kl', 0.003, 0.03)

    # Create the PPO agent with the suggested hyperparameters
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, normalize_advantage=normalize_advantage, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, target_kl=target_kl, verbose=1)

    # Train the agent for a certain number of steps
    model.learn(total_timesteps=1000)  # Adjust as needed

    # Evaluate the agent's performance
    reward_sum = 0.0
    num_episodes = 100
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

    # Return the average reward, which Optuna will try to maximize
    return reward_sum / num_episodes

def save_best_params_wrapper(save_freq):
    def save_best_params(study, trial):
        # Save the best parameters every save_freq trials
        if trial.number % save_freq == 0:
            best_params = study.best_params
            with open(f'best_params_after_{trial.number}_trials.json', 'w') as f:
                json.dump(best_params, f)
    return save_best_params

# def save_best_params(study, trial, save_freq):
#     # Save the best parameters every 100 trials
#     if trial.number % save_freq == 0:
#         best_params = study.best_params
#         with open(f'best_params_after_{trial.number}_trials.json', 'w') as f:
#             json.dump(best_params, f)
