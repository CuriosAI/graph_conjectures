import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools as it
import random
import scipy.sparse as sp

import json
import pathlib
import copy
import h5py
from collections import defaultdict
import sys
import json
import os
import pickle

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN

# from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines3.common.evaluation import evaluate_policy

# from rl_zoo3.train import train
# import optuna

# My classes
# To use custom envs with the algorithms from rl_zoo3, remember to add default hyperparameters into hyperparams file for that algorithm, e.g. rl_zoo3/hyperparams/ppo.yml

from envs import LinEnv, register_linenv#, LocEnv, GlobEnv
from graph import Graph
# from optuna_objective import objective_sb3, save_best_params_wrapper
# from tabular_mc import mc_control_epsilon_greedy, make_epsilon_greedy_policy
from save_and_load import CheckCallback, load_results, CustomExplorationScheduleCallback

##### The code starts here. This is a DQN attempt.

number_of_nodes = 4
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2

register_linenv(number_of_nodes=number_of_nodes, normalize_reward=True) # Needed by rl_zoo3. This register 'LinEnv-v0' with normalization. To change this name we need to change it also in rl_zoo3/hyperparams/ppo.yml

train_env = LinEnv(number_of_nodes, normalize_reward=True)
# number_of_states = train_env.number_of_states # Computed as 2 ** number_of_edges - 1

# Create the callback
# check_freq = number_of_edges * 1 # Check every 1 episode
check_freq = 1 # Check every 1 step
eval_env = LinEnv(number_of_nodes, normalize_reward=False) # For evaluation we don't want normalization
check_callback = CheckCallback(eval_env, check_freq=check_freq, log_file='log.txt', verbose=1)

# LinEnv is a fixed-horizon MDP. Every episode is number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2 steps long. If we want to make "similar" experiments with different number_of_nodes, it makes sense to fix the number_of_episodes instead of steps, and setting total_timesteps = number_of_edges * number_of_episodes

# number_of_episodes = 3E3 * number_of_states # For every episode we compute wagner1() of its terminal state, thus it would make sense to make episodes grow as number_of_states, but this means that total_timesteps grows as n ** 2 * 2 ** (n ** 2)!!!!

# Another option is making episodes grow as net_arch size. Bigger the network, more episodes needed to train. Does it make sense?
# policy_total_params = sum(p.numel() for p in model.policy.parameters())
# print(f"Total number of parameters in the policy network: {policy_total_params}")

# Here we set exploration for DQN to a fixed number of episodes, by setting the exploration fraction of the total timesteps of training. total_timesteps is a very high value, the idea is that training is going on until the counterexample is found. We are using RL as an exploration algorithm, we are not interested in the final policy
exploration_episodes = 5000 # To be tuned
exploration_timesteps = number_of_edges * exploration_episodes
total_timesteps = 10E9 # A large number that will never finish
exploration_fraction = exploration_timesteps / total_timesteps # Set exploration to a fixed exploration_episodes number  

exploration_final_eps = 0.05 # To be tuned
learning_rate = 1E-5 # To be tuned

# number_of_episodes = policy_total_params * 2

# total_timesteps = number_of_edges * number_of_episodes # The constant after // in number_of_episodes is chosen so that when number_of_nodes = 4, total_timesteps ~ X, where X is the number of timesteps needed to find the star at least on 5 consecutive trials when number_of_nodes = 4

# print(f"total_timesteps = {total_timesteps}")

# Create the DQN agent. net_arch = [128, 64, 4] is Wagner choice. To be tuned
net_arch = [128, 64, 4]
model = DQN('MlpPolicy', train_env, verbose=1, exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, learning_rate=learning_rate, policy_kwargs={"net_arch": net_arch}, tensorboard_log="./tensorboard_logs/")

# Train the agent until a star or a counterexample is found
model.learn(total_timesteps=total_timesteps, callback=check_callback)

# load_results("log.txt")

exit(0)


# # Test the trained agent
# state, _ = env.reset()
# for step in range(number_of_edges):
#     action, _ = model.predict(state, deterministic=True)
#     # print(f"Step {step}")
#     # print("Action: ", action)
#     state, reward, done, _, info = env.step(action)
#     # print("state=", state, "reward=", reward, "done=", done, "info", info)
#     #env.render()
#     if done:
#         # Note that the VecEnv resets automatically
#         # when a done signal is encountered
#         # print("Goal reached!", "reward=", reward)
#         # env.render()
#         graph = Graph(state[:number_of_edges])
#         print(f"\ngraph found by DQN after {total_timesteps} steps:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
#         break



# # Define the edges of the graph
# edges = [
#     (0, 6),
#     (0, 16),
#     (0, 17),
#     (1, 5),
#     (1, 8),
#     (1, 9),
#     (1, 13),
#     (1, 17),
#     (2, 4),
#     (2, 10),
#     (2, 15),
#     (2, 16),
#     (3, 4),
#     (3, 6),
#     (3, 7),
#     (3, 8),
#     (3, 9),
#     (3, 10),
#     (3, 11),
#     (3, 12),
#     (3, 14),
# ]

# # Create a new graph
# G = nx.Graph()

# # Add edges to the graph
# G.add_edges_from(edges)

# graph = Graph(G)

# print(graph.wagner1())
# graph.draw()

# plt.show()
# # Now G is a networkx graph representing the adjacency list
# exit(0)






class QDictionary:
    def __init__(self):
        pass

    def save(self, Q, filename):
        with h5py.File(filename, 'w') as hf:
            for key, value in Q.items():
                hf.create_dataset(key, data=value)

    def load(self, filename):
        Q = defaultdict(lambda: np.zeros(2))
        with h5py.File(filename, 'r') as hf:
            for key in hf.keys():
                Q[key] = np.array(hf.get(key))
        return Q

def data_collector(folder, num_nodes, connected):
  """
  Collects and reads a graph data file in .g6 (Graph6) format from a specified folder.

  Parameters:
  - folder (str): Directory path for .g6 files, taken from http://users.cecs.anu.edu.au/~bdm/data/graphs.html.
  - V (int or str): Number of nodes, used in the filename.
  - conn (bool): If True, the file .g6 contains all *connected* graphs with V nodes; otherwise, it contains all graphs with V nodes.

  Returns:
  - A NetworkX graph object from the .g6 file, containing all (connected) graphs with V nodes.
  """

  if connected == True:
    data = f"graph{num_nodes}c.g6"
  elif connected == False:
    data = f"graph{num_nodes}.g6"
  data = str(data)
  folder += data
  return nx.read_graph6(folder)


###############################
###############################
###############################
###############################
# main starts here
###############################
###############################
###############################
###############################


# Find the graphs with max wagner score by brute force, this works up to number_of_nodes = 7.
# The star is maximal between connected, the empty graph is maximal in all graphs. This info is used to check whether MC is working.

def get_max_wagner_score_graph(all_graphs):
    max_wagner_score = float('-inf')  # Initialize with negative infinity
    max_wagner_score_graph = None
    max_wagner_score_connected = float('-inf')  # Initialize with negative infinity
    max_wagner_score_graph_connected = None

    for nx_graph in all_graphs:
        graph = Graph(nx_graph)
        wagner_score = graph.wagner1()
        if wagner_score > max_wagner_score:
            max_wagner_score = wagner_score
            max_wagner_score_graph = graph
        if graph.is_connected():
            wagner_score_connected = wagner_score
            if wagner_score_connected > max_wagner_score_connected:
                max_wagner_score_connected = wagner_score_connected
                max_wagner_score_connected_graph = graph

    return max_wagner_score_graph, max_wagner_score, max_wagner_score_connected_graph, max_wagner_score_connected

def compute_min_max_wagner_scores_connected(number_of_nodes):
    number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2

    folder = "./graph_db/"
    print(f"\nloading all graphs with {number_of_nodes} nodes")
    all_graphs = data_collector(folder, number_of_nodes, False)
    print("\ndone")

    wagner_scores_connected = [Graph(nx_graph).wagner1() for nx_graph in all_graphs if nx.is_connected(nx_graph)]
    # sorted_wagner_scores = sorted(wagner_scores, key=lambda x: x, reverse=True)
    sorted_wagner_scores_connected = sorted(wagner_scores_connected)

    min_wagner_score_connected = sorted_wagner_scores_connected[0] if sorted_wagner_scores_connected else None
    max_wagner_score_connected = sorted_wagner_scores_connected[-1] if sorted_wagner_scores_connected else None

    print("\nwagner scores done")

    # print(sorted_wagner_scores_connected)

    # print(min_wagner_score_connected, max_wagner_score_connected)

    return min_wagner_score_connected, max_wagner_score_connected

# min_wagner_score_connected_list = []
# for number_of_nodes in range(3, 8):
#     min_wagner_score_connected, _ = compute_min_max_wagner_scores_connected(number_of_nodes)
#     min_wagner_score_connected_list.append(min_wagner_score_connected)

# print(min_wagner_score_connected_list)

class RewardNormalizer(gym.Wrapper):
    def __init__(self, env, min_reward):
        super().__init__(env)
        self.min_reward = abs(min_reward)

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        normalized_reward = reward / self.min_reward
        return obs, normalized_reward, done, False, info

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = self.evaluate_final_state(self.model, self.eval_env, deterministic=True, n_eval_episodes=1)
            print(f"Mean reward: {mean_reward} at step {self.n_calls}")

        return True

    def evaluate_final_state(self, model, env, n_eval_episodes=1, deterministic=True):
        rewards = []
        for _ in range(n_eval_episodes):
            state, _ = env.reset()
            done = False
            episode_rewards = 0.0

            while not done:
                action, _ = model.predict(state, deterministic=deterministic)
                state, reward, done, _, _ = env.step(action)
                episode_rewards += reward

            rewards.append(episode_rewards)
            # env.render()
            graph = Graph(state[:number_of_edges])
            print(f"graph found at step {self.n_calls}:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        return mean_reward, std_reward


#####Prova DQN


number_of_nodes = 18
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
register_linenv(number_of_nodes=number_of_nodes, normalize_reward=True) # this register 'LinEnvMau-v0', to change this name we need to change it also in rl_zoo3/hyperparams/ppo.yml
env = gym.make('LinEnvMau-v0')

total_timesteps = 100000000

# Create the callback
check_freq = 10000
eval_env = LinEnvMau(number_of_nodes, normalize_reward=False)
# callback = EvalCallback(eval_env, eval_freq=10000, verbose=1)
callback = StarCheckCallback(eval_env, check_freq=check_freq, log_file='log.txt', verbose=1)

# Create the DQN agent
model = DQN('MlpPolicy', env, verbose=1, policy_kwargs={"net_arch": [128, 64, 4]})

# print(model.policy)

# Train the agent
#model.learn(total_timesteps=50000, callback=callback)
model.learn(total_timesteps=total_timesteps, callback=callback)

# Test the trained agent
state, _ = env.reset()
for step in range(number_of_edges):
    action, _ = model.predict(state, deterministic=True)
    # print(f"Step {step}")
    # print("Action: ", action)
    state, reward, done, _, info = env.step(action)
    # print("state=", state, "reward=", reward, "done=", done, "info", info)
    #env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        # print("Goal reached!", "reward=", reward)
        # env.render()
        graph = Graph(state[:number_of_edges])
        print(f"\ngraph found by DQN after {total_timesteps} steps:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
        break

exit(0)


################## prova PPO mau

# # Check wagner1() for totally disconnected graphs
# for number_of_nodes in range(2, 20):
#   number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2

#   register_linenv(number_of_nodes)
#   env = gym.make('LinEnvMau-v0') 

#   env.reset()
#   env.render()

#   #print(env.state[:number_of_edges])
#   #env = LinEnvMau(number_of_nodes)
#   # If the environment don't follow the interface, an error will be thrown

#   # occhio che check_env modifica lo stato!!!
#   #check_env(env, warn=True)

#   #env.render()

#   graph = Graph(env.state[:number_of_edges])
#   print(graph.wagner1())


number_of_nodes = 5
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
register_linenv(number_of_nodes=number_of_nodes, normalize_reward=True) # this register 'LinEnvMau-v0', to change this name we need to change it also in rl_zoo3/hyperparams/ppo.yml

# # Load the best hyperparameters
# with open('best_params_after_7900_trials.json', 'r') as f:
#     best_params = json.load(f)
# print(best_params)

# best_params = {'learning_rate': 0.0019177642127136514, 'n_steps': 32, 'batch_size': 64, 'n_epochs': 25, 'gamma': 0.931244023824517, 'gae_lambda': 0.97327859546964, 'clip_range': 0.3, 'ent_coef': 0.004418614730618034}

# print(best_params)
# #input("Press Enter to continue...")

env = gym.make('LinEnvMau-v0')

print(env.normalize_reward)

# n_eval_episodes = 1
# rewards = []
# rewards = []
# for _ in range(n_eval_episodes):
#     state, _ = env.reset()
#     done = False
#     episode_rewards = 0.0

#     while not done:
#         action = 0
#         state, reward, done, _, _ = env.step(action)
#         episode_rewards += reward

#     rewards.append(episode_rewards)
#     # env.render()
#     graph = Graph(state[:number_of_edges])
#     print(f"graph found:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))

# mean_reward = np.mean(rewards)
# std_reward = np.std(rewards)

# print(mean_reward)

#min_reward = min_wagner_score_connected_list[number_of_nodes]
# normalized_env = RewardNormalizer(env, min_reward=-3)

# class EvalCallback(BaseCallback):
#     def __init__(self, eval_env, eval_freq, verbose=1):
#         super(EvalCallback, self).__init__(verbose)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq

#     def _on_step(self) -> bool:
#         if self.n_calls % self.eval_freq == 0:
#             mean_reward, std_reward = self.evaluate_final_state(self.model, self.eval_env, deterministic=True, n_eval_episodes=1)
#             print(f"Mean reward: {mean_reward} at step {self.n_calls}")

#         return True

#     def evaluate_final_state(self, model, env, n_eval_episodes=10, deterministic=True):
#         rewards = []
#         for _ in range(n_eval_episodes):
#             state, _ = env.reset()
#             done = False
#             episode_rewards = 0.0

#             while not done:
#                 action, _ = model.predict(state, deterministic=deterministic)
#                 state, reward, done, _, _ = env.step(action)
#                 episode_rewards += reward

#             rewards.append(episode_rewards)
#             # env.render()
#             graph = Graph(state[:number_of_edges])
#             print(f"graph found by PPO:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))

#         mean_reward = np.mean(rewards)
#         std_reward = np.std(rewards)

#         return mean_reward, std_reward

# Create the callback
# eval_env = gym.make('LinEnvMau-v0')
# callback = EvalCallback(eval_env, eval_freq=1000, verbose=1)

# train_steps = 1000000

# # Create the PPO agent with the best hyperparameters
# #model = PPO('MlpPolicy', env, **best_params, verbose=1)

# #params = {'learning_rate': 0.0019177642127136514, 'n_steps': 32, 'batch_size': 64, 'n_epochs': 25, 'gamma': 0.931244023824517, 'gae_lambda': 0.97327859546964, 'clip_range': 0.3, 'ent_coef': 0.01}

# model = PPO('MlpPolicy', normalized_env, verbose=1)

# # Train the agent
# #model.learn(total_timesteps=50000, callback=callback)
# model.learn(total_timesteps=train_steps)

# # Test the trained agent
# state, _ = env.reset()
# for step in range(number_of_edges):
#     action, _ = model.predict(state, deterministic=True)
#     # print(f"Step {step}")
#     # print("Action: ", action)
#     state, reward, done, _, info = env.step(action)
#     # print("state=", state, "reward=", reward, "done=", done, "info", info)
#     #env.render()
#     if done:
#         # Note that the VecEnv resets automatically
#         # when a done signal is encountered
#         # print("Goal reached!", "reward=", reward)
#         # env.render()
#         graph = Graph(state[:number_of_edges])
#         print(f"\ngraph found by PPO after {train_steps} steps:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
#         break

#model = PPO('MlpPolicy', 'LinEnvMau-v0')
# params = model.get_hyperparameters()
# print(params)

# # Create an Optuna study and optimize the hyperparameters
# study = optuna.create_study(direction='maximize')
# n_trials = 50000
# save_freq = 1000
# assert save_freq <= n_trials, "save_freq should be smaller or equal to n_trials"

# study.optimize(objective_sb3, n_trials=n_trials, callbacks=[save_best_params_wrapper(save_freq)])  # Adjust the number of trials as needed

# #study.optimize(objective, n_trials=n_trials)  # Adjust the number of trials as needed

# # Get the best parameters
# best_params = study.best_params

# # Save to a JSON file
# with open('best_params.json', 'w') as f:
#     json.dump(best_params, f)

# # Print the best hyperparameters
# print(f"\n\nAt the end of {n_trials} trials, the best parameters are:\n\n{best_params}. They have been saved in file best_params.json in local folder.")

# sys.argv = ["python", "--algo", "ppo", "--env", 'LinEnvMau-v0', "-n", "1000", "--optimize", "--n-trials", "50000", "--n-jobs", "56", "--sampler", "tpe", "--pruner", "median", "--progress"]


sys.argv = ["python", "--algo", "ppo", "--env", 'MiniGrid-DoorKey-5x5-v0', "--optimize", "--n-trials", "1000", "--n-jobs", "1", "--sampler", "tpe", "--pruner", "median", "--progress"]

train()

# # Load the study object
# with open('logs/ppo/report_LinEnvMau-v0_10-trials-5000-tpe-median_1689520619.pkl', 'rb') as f:
#     study = pickle.load(f)

# # Print the best value and parameters
# print(study.best_value)
# print(study.best_params)


# # Load the hyperparameters
# with open('logs/ppo/LinEnvMau-v0_ppo_hyperparams.json', 'r') as f:
#     hyperparams = json.load(f)

# print(hyperparams)


exit(0)



#sys.argv = ["python", "--algo", "ppo", "--env", "MountainCar-v0"]
sys.argv = ["python", "--algo", "ppo", "--env", 'LinEnvMau-v0', "--gym-packages", "envs"]

train()

exit(0)

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(env.action_space.n)

INSERT = 1
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2

for step in range(number_of_edges):
    print(f"Step {step}")
    obs, reward, terminated, truncated, info = env.step(INSERT)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done, "info=", info)
    #env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        # env.render()
        break

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

# Instantiate the env
vec_env = make_vec_env(LinEnvMau, n_envs=1, env_kwargs=dict(number_of_nodes=number_of_nodes))

modelppo = PPO('MlpPolicy', env, verbose=1)
modela2c = A2C("MlpPolicy", env, verbose=1)
modeldqn = DQN("MlpPolicy", env, verbose=1)

model = modelppo

# Test the untrained agent
state, _ = env.reset()
for step in range(number_of_edges):
    action, _ = model.predict(state, deterministic=True)
    print(f"Step {step}")
    print("Action: ", action)
    state, reward, done, _, info = env.step(action)
    print("state=", state, "reward=", reward, "done=", done, "info", info)
    #env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        # env.render()
        graph = Graph(state[:number_of_edges])
        print(f"graph found by PPO with random weights:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
        break

# Train the agent
train_steps = 100000
print(f"\nTraining the PPO agent for {train_steps} steps\n")
model.learn(train_steps)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False, deterministic=True)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test the trained agent
state, _ = env.reset()
for step in range(number_of_edges):
    action, _ = model.predict(state, deterministic=True)
    print(f"Step {step}")
    print("Action: ", action)
    state, reward, done, _, info = env.step(action)
    print("state=", state, "reward=", reward, "done=", done, "info", info)
    #env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        # env.render()
        graph = Graph(state[:number_of_edges])
        print(f"\ngraph found by PPO after {train_steps} steps:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
        break
    
    


######################### prova MC MAU

print("\nTabular MC test\n")

#number_of_nodes = 5
#number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
#num_episodes = 12 * (2 ** (number_of_edges +1) - 1)
num_episodes = 1000
env = LinEnvMau(number_of_nodes)
# state = env.reset()
#print(env.state)

# Exponential learning rate schedule for tabular MC: a list with episode numbers,
# when those episode numbers are reached, lr is divided by 2
#schedule=[2**n for n in range(10, int(np.log2(num_episodes)) + 1)]

# Linear lr schedule
lr_cuts = 4
schedule=[n for n in range(num_episodes // lr_cuts, num_episodes, num_episodes // lr_cuts)]

print(schedule)

filename = f'Q_dict_{number_of_nodes}.h5'
Q_dict = QDictionary()
Q = defaultdict(lambda: np.zeros(2))
ct = 0

# Load Q if it was previously written, look for a filename f'Q_dict_{number_of_nodes}.h5'
if not os.path.exists(filename):
  print(f"\nfirst start of MC, Q is empty\n")
  print(f"\nstarting MC for the first time with an empty Q for {num_episodes} episodes\n")
  Q, policy, episodes, Q_diff_norms = mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.1, epsilon=1, schedule=schedule, max_steps=1000000, save=True)
  print(f"\nfirst MC done, now Q contains {len(Q)} states\n")
  # Save the dictionary
  Q_dict.save(Q, filename)
  print("\nnew Q saved\n")
  ct = ct + 1
  print(f"\n{ct * num_episodes} initial episodes done, saving each {num_episodes} episodes\n")
  
  # # find and print the optimal graph up to now
  # greedy_policy = make_epsilon_greedy_policy(Q=Q, epsilon=0.0, nA=env.nA)
  # env.reset()
  # while not env.done:
  #   probs = greedy_policy(env.state)
  #   action = np.random.choice(np.arange(len(probs)), p=probs)
  #   #print(env.state)
  #   #print(action)
  #   state, reward, done = env.step(action)
  #   #print(f"after action {action} we get state = {state}, reward = {reward}, done = {done}")
  #   final_state = copy.deepcopy(env.state)
  # graph = Graph(final_state[:number_of_edges])
  # print(f"graph found by the greedy policy after {ct * num_episodes} episodes:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))

# MC training in batches of num_episodes, with lr schedule resetting after every batch
while True:
  # Find and print the optimal graph with the Q-greedy policy
  greedy_policy = make_epsilon_greedy_policy(Q=Q, epsilon=0.0, n=env.action_space.n)
  env.reset()
  while not env.done:
    probs = greedy_policy(env.state)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    #print(env.state)
    #print(action)
    state, reward, done, _, _ = env.step(action)
    #print(f"after action {action} we get state = {state}, reward = {reward}, done = {done}")
    final_state = copy.deepcopy(env.state)
  graph = Graph(final_state[:number_of_edges])
  print(f"\ngraph found by the greedy policy after {ct * num_episodes} episodes in this last run:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))
  # Load Q
  Q = Q_dict.load(filename)
  print(f"\nold Q contained {len(Q)} states\n")
  print(f"\nstarting MC\n")
  Q, policy, episodes, Q_diff_norms = mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.1, epsilon=1, schedule=schedule, max_steps=1000000, Q=Q, save=True)
  print(f"\nafter batch #{ct+1} of {num_episodes} MC episodes, Q contains {len(Q)} states")
  # Save the dictionary
  Q_dict.save(Q, filename)
  print("\nnew Q saved")
  ct = ct + 1
  print(f"{ct * num_episodes} episodes done in this 'while True' loop, saving each {num_episodes} episodes")
  




exit(0)

# Find the graphs with max wagner score by brute force, this works up to number_of_nodes = 7.
# The star is maximal between connected, the empty graph is maximal in all graphs. This info is
# used to check whether MC is working.

# def get_max_wagner_score_graph(all_graphs):
#     max_wagner_score = float('-inf')  # Initialize with negative infinity
#     max_wagner_score_graph = None
#     max_wagner_score_connected = float('-inf')  # Initialize with negative infinity
#     max_wagner_score_graph_connected = None

#     for nx_graph in all_graphs:
#         graph = Graph(nx_graph)
#         wagner_score = graph.wagner1()
#         if wagner_score > max_wagner_score:
#             max_wagner_score = wagner_score
#             max_wagner_score_graph = graph
#         if graph.is_connected():
#             wagner_score_connected = wagner_score
#             if wagner_score_connected > max_wagner_score_connected:
#                 max_wagner_score_connected = wagner_score_connected
#                 max_wagner_score_connected_graph = graph

#     return max_wagner_score_graph, max_wagner_score, max_wagner_score_connected_graph, max_wagner_score_connected

# folder = "./graph_db/"
# print(f"\nloading all graphs with {number_of_nodes} nodes")
# all_graphs = data_collector(folder, number_of_nodes, False)
# print("\nloaded")

# print(len(all_graphs))

# wagner_scores = [(Graph(nx_graph).wagner1(), Graph(nx_graph).is_connected()) for nx_graph in all_graphs]
# sorted_wagner_scores = sorted(wagner_scores, key=lambda x: x[0], reverse=True)

# max_wagner_score_graph, max_wagner_score, max_wagner_score_connected_graph, max_wagner_score_connected = get_max_wagner_score_graph(all_graphs)

# print("\nwagner scores done")

# print(sorted_wagner_scores)

# print(max_wagner_score, max_wagner_score_connected)


# # Create a figure and axes
# fig, ax = plt.subplots()
# pos = nx.spring_layout(max_wagner_score_graph.graph)
# ax.clear()
# # Draw the new graph
# for graph in [max_wagner_score_graph, max_wagner_score_connected_graph]:
#   plt.title(f"wagner1 score = {graph.wagner1()}")
#   nx.draw(graph.graph, pos=pos, ax=ax, with_labels=True)
#   # Update the display
#   plt.draw()
#   # Pause for a moment to show the plot
#   plt.pause(1)
# # Keep the window open
# #plt.show()

# counter_examples_indexes = []
# for i, nx_graph in enumerate(all_graphs):
#     if i % 1000 == 0:
#       print(f"\rgraph {i}/{len(all_graphs)}", end="")
#       sys.stdout.flush()

#     graph = Graph(nx_graph)
#     if graph.is_connected():
#         if graph.wagner1() > 0:
#             print(f"Condition does not hold for graph #{i}")
#             counter_examples_indexes.append(i)
#             # const = 1 + np.sqrt(graph.num_nodes - 1)
#             # radius = max(np.real(nx.adjacency_spectrum(graph.graph)))
#             # weight = len(nx.max_weight_matching(graph.graph))
#             # wagner1_score = const - (radius + weight)

#             # # Create a figure and axes
#             # fig, ax = plt.subplots()
#             # pos = nx.spring_layout(graph.graph)
#             # ax.clear()
#             # # Draw the new graph
#             # plt.title(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")
#             # nx.draw(graph.graph, pos=pos, ax=ax, with_labels=True)
#             # # Update the display
#             # plt.draw()
#             # # Pause for a moment to show the plot
#             # plt.pause(1)
#             # # Keep the window open
#             # #plt.show()
#             # break
# # else:
# #     print("Condition holds for all connected graphs.")

# print("\n\n", counter_examples_indexes)

# print(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")

# graph = Graph(all_graphs[8])
# const = 1 + np.sqrt(graph.num_nodes - 1)
# radius = max(np.real(nx.adjacency_spectrum(graph.graph)))
# weight = len(nx.max_weight_matching(graph.graph))

# print(nx.adjacency_matrix(graph.graph).toarray())
# print(nx.adjacency_spectrum(graph.graph))

# print(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")

# def show_adjacency(Q):
#   # find and print the optimal graph up to now
#   print("printing the optimal graph...")
#   greedy_policy = make_epsilon_greedy_policy(Q=Q, epsilon=0.0, nA=env.nA)
#   env.reset()
#   while not env.done:
#     probs = greedy_policy(env.state)
#     action = np.random.choice(np.arange(len(probs)), p=probs)
#     #print(env.state)
#     #print(action)
#     state, reward, done = env.step(action)
#     #print(f"after action {action} we get state = {state}, reward = {reward}, done = {done}")
#     final_state = copy.deepcopy(env.state)
#   graph = Graph(final_state[:number_of_edges])
#   print(f"graph found by the greedy policy after {ct * num_episodes} episodes:\n", sp.triu(nx.adjacency_matrix(graph.graph), format='csr'))




states = [state for episode in episodes for state, _, _, _ in episode]
#print("states done")
next_states = [next_state for episode in episodes for _, _, _, next_state in episode]
#print("next states done")
all = states + next_states # all states visited during episodes, including final states
all = [tuple(s) for s in all] # later we need it immutable
#print("all states done")

assert len(all) == num_episodes * number_of_edges * 2

all_unique = list(dict.fromkeys(all))

#assert len(all_unique) == 2 ** (number_of_edges +1) -1
print(f"\n\nexploration: {len(all_unique)} of {2 ** (number_of_edges +1) -1}")

# frequencies = []

# for state in all_unique:
#   frequencies.append(all.count(state) / len(all))

# print(sum(frequencies))
# print(1/len(frequencies), np.min(frequencies), np.max(frequencies))

# from scipy.special import kl_div

# def relative_entropy(p):
#     # Calculate the uniform distribution
#     uniform_p = np.ones(len(p)) / len(p)
    
#     # Calculate the Kullback-Leibler divergence
#     kl_divergence = kl_div(p, uniform_p)
    
#     return np.sum(kl_divergence)

# print(relative_entropy(frequencies))

greedy_policy = make_epsilon_greedy_policy(Q=Q, epsilon=0.0, nA=env.nA)
env.reset()
while not env.done:
  probs = greedy_policy(env.state)
  action = np.random.choice(np.arange(len(probs)), p=probs)
  #print(env.state)
  #print(action)
  state, reward, done = env.step(action)
  #print(f"after action {action} we get state = {state}, reward = {reward}, done = {done}")
  final_state = copy.deepcopy(env.state)

#print(final_state)

graph = Graph(final_state[:number_of_edges])

#max_wagner_score_graph.draw()
graph.draw()
print(graph.wagner1())

# # Assume Q_diff_norms is the list of norms you got from your function

# plt.figure(figsize=(10,6))
# plt.plot(Q_diff_norms)
# plt.title('Norm of Q-value Differences Over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Norm of Q-value Differences')
# plt.show()

print(nx.adjacency_matrix(graph.graph))

plt.show()
exit(0)

# Sort the states based on the timestep part
all_unique.sort(key=lambda s: (s[-6:],s[:6]))

ct = 0
# Print the sorted states
for state in all_unique:
    print(ct,"--------------->",state)
    ct+=1




exit(0)
set[states]


count = sum(action == 1 for episode in episodes for _, action, _, _ in episode)
count = sum(action == 1 for episode in episodes for _, action, _, _ in episode)
total_actions = sum(len(episode) for episode in episodes)

print(count, total_actions, count/total_actions)

# Convert the keys of the Q dictionary to a list
states = list(Q.keys())

exit(0)


V = 4
E = int((V*(V-1))//2)
Env = LinEnv(V)
Env.reset()

constant_policy = lambda x: torch.Tensor([0.5])

# player = Agent(Env, wagner, l_r=0.0001)
# player.reset()

player = Agent(Env, constant_policy, l_r=0.0001)
player.reset()

# Sample a trajectory
sts, acts, graphs, final_graph, rews = player.policy_sample()
print(acts)
print(len(graphs))
graphs.append(final_graph)
print(len(graphs))

# Visualize the graphs in the trajectory using networkx

# Create a figure and axes
fig, ax = plt.subplots()

for i, graph in enumerate(graphs):
   
    # fix positions of nodes for all draws
    if i==0:
      pos = nx.spring_layout(graph)

    # Clear the current plot
    ax.clear()
    
    # Draw the new graph
    plt.title(f"Graph at step {i}")
    nx.draw(graph, pos=pos, ax=ax, with_labels=True)
    
    # Update the display
    plt.draw()
    
    # Pause for a moment to show the plot
    plt.pause(1)

# Keep the window open
plt.show()

