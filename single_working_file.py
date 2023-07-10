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

import json
import pathlib
import copy
import h5py
from collections import defaultdict
import sys

# my classes
# from feats_miner import feats_miner
from envs import LinEnvMau#, LocEnv, GlobEnv
from graph import Graph
from tabular_mc import mc_control_epsilon_greedy, make_epsilon_greedy_policy

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


######################### prova MC MAU

number_of_nodes = 9
number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
#num_episodes = 12 * (2 ** (number_of_edges +1) - 1)
num_episodes = 5000
env = LinEnvMau(number_of_nodes)
# state = env.reset()
#print(env.state)

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

folder = "./graph_db/"
print(f"\nloading all graphs with {number_of_nodes} nodes")
all_graphs = data_collector(folder, number_of_nodes, False)
print("\nloaded")

print(len(all_graphs))
#print(type(all_graphs[0]))

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

schedule=[2**n for n in range(10, int(np.log2(num_episodes)))]
#print(schedule)

print("\nstarted MC\n")
Q, policy, episodes, Q_diff_norms = mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.1, epsilon=1, schedule=schedule, max_steps=1000000, save=True)

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

