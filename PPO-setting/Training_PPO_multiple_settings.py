import os
import csv
import time
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from envs.linear_environment import LinearEnvironment
from envs.local_environment import LocalEnvironment
from envs.global_environment import GlobalEnvironment
import numpy as np
import networkx as nx
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

def get_action_mask(env):
    return env.unwrapped.get_valid_actions()

class RewardStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ep_rewards = self.model.ep_info_buffer

        if len(ep_rewards) > 0:
            rewards = [ep['r'] for ep in ep_rewards]
            self.logger.record("rollout/ep_rew_max", np.max(rewards))
            self.logger.record("rollout/ep_rew_min", np.min(rewards))

class RolloutConvergenceCallback(BaseCallback):
    def __init__(self, rollout_csv_path, avg_threshold, var_threshold, window_size=5, verbose=1):
        super(RolloutConvergenceCallback, self).__init__(verbose)
        self.rollout_csv_path = rollout_csv_path
        self.avg_threshold = avg_threshold
        self.var_threshold = var_threshold
        self.window_size = window_size
        self.last_row_read = 0
        self.verbose = 1
        self.rolling_rewards = []
        if not os.path.exists(self.rollout_csv_path):
            with open(self.rollout_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["time/total_timesteps", "rollout/ep_rew_mean", "rollout/ep_rew_max", "rollout/ep_rew_min"])

    def _on_step(self) -> bool:
        if not os.path.exists(self.rollout_csv_path):
            if self.verbose > 0:
                print(f"File {self.rollout_csv_path} not yet created.")
            return True

        try:

            rollout_data = pd.read_csv(self.rollout_csv_path)
            
            if "rollout/ep_rew_mean" in rollout_data.columns:
                if len(rollout_data) > self.last_row_read:
                    current_reward = rollout_data["rollout/ep_rew_mean"].iloc[-1]
                    self.last_row_read = len(rollout_data)
            

                    self.rolling_rewards.append(current_reward)
                    if len(self.rolling_rewards) > self.window_size:
                        self.rolling_rewards.pop(0)

                    if len(self.rolling_rewards) == self.window_size:
                        rolling_avg = sum(self.rolling_rewards) / self.window_size
                        rolling_var = pd.Series(self.rolling_rewards).var()
                        avg_change = self.rolling_rewards[-1] - rolling_avg

                        if self.verbose > 0:
                            print(
                                f"Rolling Avg: {rolling_avg:.4f}, Variance: {rolling_var:.4f}, "
                                f"Avg Change: {avg_change:.4f}"
                            )

                        if avg_change < self.avg_threshold and avg_change >=0 and rolling_var < self.var_threshold:
                            print(f"Convergence reached: Avg Change {avg_change:.6f} < {self.avg_threshold}, "
                                  f"Variance {rolling_var:.6f} < {self.var_threshold}")
                            return False

        except Exception as e:
            if self.verbose > 0:
                print(f"Error reading rollout CSV: {e}")

        return True


def train_with_convergence(env, model_mode, size, total_timesteps, learning_rate, entropy_level, gamma, gae_lambda, batch_size, log_dir, rollout_csv, avg_threshold, var_threshold, policy_kwargs):

    logger = configure(log_dir, ["stdout", "csv"])
    monitor_env = Monitor(env)

    model = MaskablePPO("MlpPolicy", monitor_env, 
                gamma = gamma, 
                gae_lambda = gae_lambda,
                learning_rate = learning_rate,
                ent_coef = entropy_level, 
                batch_size = batch_size, 
                policy_kwargs = policy_kwargs,
                verbose = 1)
    model.set_logger(logger)

    callback = RolloutConvergenceCallback(
        rollout_csv_path=rollout_csv,
        avg_threshold=avg_threshold,
        var_threshold=var_threshold,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps, callback=[callback, RewardStatsCallback()], progress_bar=True)
    model.save(f"{log_dir}/ppo_{size}_{model_mode}")
    logger.close()

def wagner_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    if nx.is_connected(g):
        info = {'connected'}
        const = 1 + np.sqrt(n - 1)
        radius = max(np.real(nx.adjacency_spectrum(g)))
        weight = len(nx.max_weight_matching(g))
        wagner = const - (radius + weight)
        reward = wagner/n if normalize else wagner
    else:
        reward = -2*n if normalize else -4*n  
        info = {'not_connected'}
        
    return reward

def brouwer_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1,n+1)])
    diff = sums - (binomials + m)
    if normalize:
        return np.tanh(max(diff[2:n-2]))
    else:
        return max(diff[2:n-2])
    
def central_brouwer_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1,n+1)])
    diff = sums - (binomials + m)
    lower = int(0.25*n)
    upper = int(0.75*n) - 1 
    if normalize:
        return np.tanh(max(diff[lower:upper]))
    else:
        return max(diff[lower:upper])

if __name__ == "__main__":
    node_sizes = [11,12,13,14,15]
    model_modes = ["complete", "central"]

    env_classes = {
        "Linear": LinearEnvironment,
        "Local": LocalEnvironment,
        "Global": GlobalEnvironment,
    }
    total_timesteps = 200000
    learning_rate = 0.0003
    batch_size = 2048
    entropy = 0.01

    avg_threshold = 0.001
    var_threshold = 0.0005
    gamma = 0.99
    gae_lambda = 0.95
    policy_kwargs=dict(net_arch=[128, 128])
    for model_mode in model_modes:
        for size in node_sizes:
            for env_key, EnvClass in env_classes.items():
                print(f"\n--- Training {env_key}Environment with {size} nodes ---")
                try:
                    if model_mode == "central":
                        env = EnvClass(size, central_brouwer_score, verbose=0, normalize_reward=False)
                        env_name = env.name
                    elif model_mode == "complete":
                        env = EnvClass(size, brouwer_score, verbose=0, normalize_reward=False)
                        env_name = env.name
                    model_name = f"PPO_{size}_{model_mode}"
                    env = ActionMasker(env, get_action_mask)
                    
                    log_dir = f"./models/{env_name}"
                    os.makedirs(log_dir, exist_ok=True)
                    rollout_csv = f"{log_dir}/progress.csv"

                    train_with_convergence(env, 
                                        size = size,
                                        model_mode = model_mode,
                                        total_timesteps=total_timesteps,
                                        learning_rate=learning_rate,
                                        entropy_level=entropy,
                                        gamma=gamma,
                                        gae_lambda=gae_lambda,
                                        batch_size=batch_size,
                                        log_dir=log_dir,
                                        rollout_csv=rollout_csv,
                                        avg_threshold=avg_threshold,
                                        var_threshold=var_threshold,
                                        policy_kwargs = policy_kwargs)
                    time.sleep(1)
                    final_csv = f"{log_dir}/progress_{size}_{model_mode}.csv"
                    if os.path.exists(rollout_csv):
                        os.replace(rollout_csv, final_csv)
                except Exception as e:
                    print(f"Errore durante il training per {env_key} con size {size}: {e}")