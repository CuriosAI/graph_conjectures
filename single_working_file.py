# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as functionals
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from torchsummary import summary

# import torch_geometric
# from torch_geometric.utils import from_networkx

# import gymnasium as gym

# from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.metrics import log_loss, mean_squared_error

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools as it
import random
import scipy.sparse as sp
# import keyboard

import json
import pathlib
import copy
import h5py
from collections import defaultdict
import sys
import json
import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rl_zoo3.train import train
import optuna


# my classes
# from feats_miner import feats_miner
from envs import LinEnvMau, register_linenv#, LocEnv, GlobEnv
from graph import Graph
from optuna_objective import objective, save_best_params_wrapper
from tabular_mc import mc_control_epsilon_greedy, make_epsilon_greedy_policy

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


number_of_nodes = 6
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
register_linenv(number_of_nodes)

# Load the best hyperparameters
with open('best_params_after_7900_trials.json', 'r') as f:
    best_params = json.load(f)

env = gym.make('LinEnvMau-v0')

# Create the PPO agent with the best hyperparameters
model = PPO('MlpPolicy', env, **best_params, verbose=0)

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, deterministic=True, n_eval_episodes=1)
            print(f"Mean reward: {mean_reward} +/- {std_reward}")

        return True

# Create the callback
callback = EvalCallback(env, eval_freq=1000, verbose=1)

# Train the agent
model.learn(total_timesteps=50000, callback=callback)


exit(0)

#model = PPO('MlpPolicy', 'LinEnvMau-v0')
# params = model.get_hyperparameters()
# print(params)



# Create an Optuna study and optimize the hyperparameters
study = optuna.create_study(direction='maximize')
n_trials = 100000
save_freq = 100
assert save_freq <= n_trials, "save_freq should be smaller or equal to n_trials"

study.optimize(objective, n_trials=n_trials, callbacks=[save_best_params_wrapper(save_freq)])  # Adjust the number of trials as needed

#study.optimize(objective, n_trials=n_trials)  # Adjust the number of trials as needed

# Get the best parameters
best_params = study.best_params

# # Save to a JSON file
# with open('best_params.json', 'w') as f:
#     json.dump(best_params, f)



# Print the best hyperparameters
print(f"\n\nAt the end of {n_trials} trials, the best parameters are:\n\n{best_params}. They have been saved in file best_params.json in local folder.")

# sys.argv = ["python", "--algo", "ppo", "--env", 'LinEnvMau-v0']

# train()

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

