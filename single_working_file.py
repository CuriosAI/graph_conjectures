# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as functionals
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from torchsummary import summary

# import torch_geometric
# from torch_geometric.utils import from_networkx

import gymnasium as gym

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error

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
from feats_miner import feats_miner
from envs import LinEnvMau, LocEnv, GlobEnv
from graph import Graph
from tabular_mc import mc_control_epsilon_greedy



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

def dict_to_arr(dict):
  """
  Converts a dictionary into a numpy array.

  Parameters:
  - dict (dictionary): The dictionary to be converted. Keys are expected to be integers from 0 to len(dict)-1.

  Returns:
  - X (numpy array): A column vector where the i-th element is the value of the i-th key in the dictionary.
  """

  v = len(dict)
  X = np.zeros((v,1))
  for i in range(v):
    X[i,0] += dict[i]
  return X

def cut_nodes(G):
  """
  Creates a list of graphs, each one being a copy of the original graph with one node removed.

  Parameters:
  - G (NetworkX graph): The original graph.

  Returns:
  - cuts (list): A list of graphs. Each graph in the list is a copy of the original graph with one node removed.
  """

  v = G.number_of_nodes()
  cuts = [copy.deepcopy(G) for _ in range(v)]
  for i in range(v):
    cuts[i].remove_node(i)
  return cuts


path_exp = './experiments/'

class ExperimentData:
  def __init__(self, target_score, path=path_exp):
    self.path = pathlib.Path(path + target_score)
    self.path.mkdir(parents=True, exist_ok=True)
    self.data_path = None
    self.groups_path = {}
    self.subgroups_dep = {}

  def load_database(self, env_name):
    self.data_path = self.path / (env_name + '.hdf5')
    try:
      with h5py.File(self.data_path, 'a') as file:
        for group_name in file.keys():
          self.groups_path[group_name] = str(self.data_path / group_name)
          for subgroup_name in file[group_name].keys():
            self.subgroups_dep[subgroup_name] = group_name
    except Exception as e:
      print(f"Failed to create HDF5 file at {self.data_path}: {e}")

  def add_group(self, group_name, meta=None):
    try:
      with h5py.File(self.data_path, 'a') as file:
        grp = file.create_group(group_name)
        if meta is not None:
          for key, value in meta.items():
            file[group_name].attrs[key] = value
        self.groups_path[group_name] = str(self.data_path / group_name)
    except ValueError:
      print(f"Group '{group_name}' already exists.")
    except Exception as e:
      print(f"Failed to create group {group_name} in HDF5 file at {self.data_path}: {e}")

  def get_group_meta(self, env):
    meta = {}
    meta['num_nodes'] = env.num_nodes
    meta['rew_type'] = env.rew_type
    return meta

  def up_group_meta(self, group_name, meta):
    try:
      with h5py.File(self.data_path, 'a') as file:
        if group_name in file:
          for key, value in meta.items():
            file[group_name].attrs[key] = value
        else:
          print(f"The group '{group_name}' does not exist in the file.")
    except Exception as e:
        print(f"Failed to open HDF5 file at {self.data_path}: {e}")

  def add_subgroup(self, parent_group, subgroup_name, meta=None):
    if subgroup_name not in self.subgroups_dep:
      self.subgroups_dep[subgroup_name] = parent_group
    try:
      with h5py.File(self.data_path, 'a') as file:
        parent_group = file[parent_group]
        subgroup = parent_group.create_group(subgroup_name)
        if meta is not None:
          for key, value in meta.items():
            file[parent_group][subgroup_name].attrs[key] = value
    except ValueError:
      print(f"Subgroup '{subgroup_name}' already exists.")
    except Exception as e:
      print(f"Failed to create subgroup {subgroup_name} in HDF5 file at {self.data_path}: {e}")

  def add_simulation(self, subgroup_name, sim_name, simulation, meta=None):
    parent = self.subgroups_dep[subgroup_name]
    adjs, sarx = simulation
    try:
      with h5py.File(self.data_path, 'a') as file:
        subgroup = file[parent + '/' + subgroup_name]
        datasets = list(subgroup.keys())
        if any(ds.startswith(sim_name) for ds in datasets):
          print(f"Simulation name '{sim_name}' already taken.")
        else:
          subgroup.create_dataset(sim_name + '_adjs', data=adjs)
          subgroup.create_dataset(sim_name + '_sarx', data=sarx)
          if meta is not None:
            for key, value in meta.items():
              subgroup[sim_name + '_adjs'].attrs[key] = value
              subgroup[sim_name + '_sarx'].attrs[key] = value
    except Exception as e:
      print(e)

class Agent():
    """
    The Agent class is responsible for training a policy network in an environment.
    The policy network is used to make decisions in the environment, and the agent
    learns from the outcomes to improve the policy over time.

    Attributes:
        policy (nn.Module): The neural network policy used to make decisions.
        MAU: la policy si assume che sia una rete neurale, perché nell'init si usa policy.parameters, dovrebbe
        essere una qualunque funzione che prende stati e restituisce probabilità.
        env (object): The environment in which the agent operates.
        learning_rate (float): The learning rate used for training the policy.
        criterion (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        objective (int): The current objective score that the agent is trying to maximize.
        baseline (int): The baseline score used to determine if the agent is improving.
        success_hist (list): A history of successful outcomes.
        loss_hist (list): A history of loss values during training.

    Methods:
        reset(): Resets the agent's state.
        policy_sample(): Samples an episode using the current policy.
        sample_process(sample): Processes a sample episode to extract valid states and rewards.
        one_shot_training(num_steps, num_eps): Trains the policy network for a given number of steps and episodes.
    """

    def __init__(self, Env, Net, l_r, conn_check=True):
      self.policy = Net
      self.env = Env
      self.learning_rate = l_r
      #self.criterion = torch.nn.BCELoss()
      #self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.learning_rate)
      # MAU: commentati, perché Agent non deve allenare, va tolto anche il metodo one_shot_training.
      self.objective = 0
      self.baseline = -100
      self.success_hist = []
      self.loss_hist = []


    def reset(self):
      self.objective = 0
      self.baseline = -100
      self.success_hist = []
      self.loss_hist = []
      return

    def policy_sample(self):
      """
      Samples an episode from the environment using the current policy.
      The method resets the environment and then takes actions according to
      the policy until the environment is done. It keeps track of the states,
      actions, graphs, and rewards at each step. Finally, it returns the states,
      actions, graphs, final graph, and rewards of the entire episode.

      Returns:
          tuple: A tuple containing the following elements:
              - sts (list): A list of tensors representing the states at each step.
              - acts (list): A list of integers representing the actions taken at each step.
              - graphs (list): A list of graphs representing the state of the graph at each step.
              - final_graph (networkx.Graph): The final graph at the end of the episode.
              - rews (list): A list of rewards obtained at each step.

      Example:
          sts, acts, graphs, final_graph, rews = agent.policy_sample()
      """

      self.env.reset()
      sts = []
      acts = []
      graphs = []
      while not self.env.done:
        sts.append(torch.Tensor(copy.deepcopy(self.env.state)))
        act = int(torch.bernoulli(self.policy(sts[-1])).item())
        acts.append(act)
        graphs.append(copy.deepcopy(self.env.graph))
        self.env.step(act)
      rews = copy.deepcopy(self.env.score_traj)
      final_graph = copy.deepcopy(self.env.graph)
      return sts, acts, graphs, final_graph, rews

    def sample_process(self,sample):
      """
      Processes a sample from the environment and returns relevant information
      if the sample is valid. The method checks if the number of connected components
      in the final graph is equal to 1. If not, it returns None. Otherwise, it computes
      the rewards for valid graphs and returns the tensor of states, tensor of actions,
      value, final graph, and the number of edges in the environment.

      Parameters:
          sample (tuple): A tuple containing the states, actions, graphs, final graph,
                          and rewards from the environment.

      Returns:
          tuple or None: Returns a tuple containing the tensor of states, tensor of actions,
                        value, final graph, and the number of edges in the environment if
                        the sample is valid. Otherwise, returns None.

      Example:
          sample = (states, actions, graphs, final_graph, rewards)
          processed_sample = agent.sample_process(sample)
      """

      valid_rews = []
      if self.env.miner.invariants['num_cc'](sample[3]) != 1:
        return None
      else:
        final_rew = sample[4][-1]
        for p in range(self.env.num_edges-1,-1,-1):
          if self.env.miner.invariants['num_cc'](sample[2][p]) == 1:
            valid_rews.append(sample[4][p])
          else:
            break
        if valid_rews == [] or (final_rew > max(valid_rews)):
          Tens_sts = torch.stack(sample[0])
          Tens_acts = torch.Tensor(sample[1])
          value = final_rew
          graph = sample[3]
          return Tens_sts, Tens_acts, value, graph, self.env.num_edges
        else:
          num_valid_graphs = len(valid_rews)
          valid_rews.reverse()
          value = max(valid_rews)
          t = valid_rews.index(value)
          t += self.env.num_edges - num_valid_graphs
          graph = sample[2][t]
          Tens_sts = torch.stack(sample[0])
          for l in range(t+1,self.env.num_edges):
            Tens_sts[l,:self.env.num_edges] = Tens_sts[t,:self.env.num_edges]
          Tens_acts = Tens_sts[-1,:self.env.num_edges]
          return Tens_sts, Tens_acts, value, graph, t

    def one_shot_training(self,num_steps,num_eps):
      steps = num_steps
      num_trains = 0
      while steps > 0:
        curr_objective = copy.deepcopy(self.baseline)
        Y = self.policy_sample()
        X = self.sample_process(Y)
        if X == None:
          print("Invalid game")
        else:
          print(f"Current best score: {X[2]}")
          if X[2] > self.objective:
            print(f"Objective achieved in {num_steps-steps} rounds by:")
            # nx.draw(X[3])
            # plt.show()
            # plt.close()
            self.success_hist += [X[3]]
            print(f"at position {X[4]}")
            for t in range(num_eps):
              preds = self.policy(X[0]).squeeze()
              L = self.criterion(preds,X[1])

              self.optimizer.zero_grad()
              L.backward()
              self.optimizer.step()
              self.loss_hist.append(L.item())
              if t == num_eps-1:
                print(f"Loss: {L}")
            self.baseline = copy.deepcopy(self.objective)
            self.objective = X[2]
            return X
          elif X[2] > curr_objective:
            print("Current task passed, learning from:")
            # nx.draw(X[3])
            # plt.show()
            # plt.close()
            self.success_hist += [X[3]]
            print(f"at position {X[4]}")
            for t in range(num_eps):
              preds = self.policy(X[0]).squeeze()
              L = self.criterion(preds,X[1])

              self.optimizer.zero_grad()
              L.backward()
              self.optimizer.step()
              if t == num_eps-1:
                print(f"Loss: {L}")
                self.loss_hist.append(L.item())
            self.baseline = X[2]
            print(f"current objective: {self.baseline}")
        steps -= 1
      return self.baseline, self.loss_hist, self.success_hist

def sess_gen(Env,Net,n_gen,conn=True):
  P = Env
  Eps = []
  for i in range(n_gen):
    P.reset()
    sts = []
    acts = []
    graphs = []
    while not P.done:
      sts.append(torch.Tensor(copy.deepcopy(P.state)))
      act = int(torch.bernoulli(Net(sts[-1])).item())
      acts.append(act)
      graphs.append(copy.deepcopy(P.graph))
      P.step(act)
    rews = copy.deepcopy(P.score_traj)
    co_adj = torch.stack(sts)
    co_adj_embedded = []
    if P.miner.invariants['num_cc'](graphs[-1]) == 1:
      co_adj_embedded = [co_adj]
      for p in range(P.num_edges-2,-1,-1):
        if P.miner.invariants['num_cc'](graphs[p]) == 1:
          co_adj_embedded.append(co_adj)
        else:
          break
    if co_adj_embedded != []:
      co_adj_tens = torch.stack(co_adj_embedded)
      for q in range(1,co_adj_tens.shape[0]):
        for r in range(Env.num_edges-q,Env.num_edges):
          for s in range(Env.num_edges-q,Env.num_edges):
            co_adj_tens[q,r,s] = 0
      invs = rews[Env.num_edges-co_adj_tens.shape[0]-1:]
      contr_ind = invs.index(max(invs))
      return co_adj_tens.shape, len(invs)



    if conn == True:
      stop = P.num_edges

      if P.miner.invariants['num_cc'](graph) == 1:
        Eps.append((sts,acts,rew,graph))
    else:
      Eps.append((sts,acts,rew,graph))
  return Eps

def best_individuals(sesss, cap, duplicates=False):
  if sesss == []:
    return sesss
  sorted_lex = sorted(sesss, key=lambda x: x[1])
  repr = [sorted_lex[0]]
  if duplicates == False:
    for i in range(1,len(sorted_lex)):
      if sorted_lex[i][1] != repr[-1][1]:
        repr.append(sorted_lex[i])
  else:
    repr = sorted_lex
  sorted_repr = sorted(repr, key=lambda x: x[2], reverse=True)
  return sorted_repr[:cap]



def brouwer_vector(G):
  spec = nx.laplacian_spectrum(G)
  V = len(spec)
  B = []
  eig_sum = 0
  for t in range(V-1,-1,-1):
    eig_sum += spec[t]
    m = G.number_of_edges()
    c = math.comb(V+1-t,2)
    score = eig_sum - (m+c)
    B.append(score)
  return B

def edge_class(G):
  e = nx.number_of_edges(G)
  V = nx.number_of_nodes(G)
  E = int((V*(V-1))//2)
  H = torch.zeros((1,E))
  H[0,e-1] += 1
  return H

def tensor_data(L,size):
  l = len(L)
  card = int(l*size)
  S = random.sample(L, card)
  T_in = []
  T_out = []
  for G in S:
    T_in.append(nx.laplacian_spectrum(G))
    T_out.append(edge_class(G))
  T_in = np.array(T_in)
  T_in = torch.Tensor(T_in)
  T_out = torch.cat(T_out, dim=0)
  dataset = TensorDataset(T_in, T_out)
  return T_in, T_out, dataset

def arr_data_cc(L,size):
  l = len(L)
  card = int(l*size)
  S = random.sample(L,card)
  T_in = []
  T_out = []
  for G in S:
    T_in.append(nx.laplacian_spectrum(G))
    T_out.append(nx.number_connected_components(G))
  T_in = np.array(T_in)
  T_out = np.array(T_out)
  return T_in, T_out

def arr_data_ed(L,size):
  l = len(L)
  card = int(l*size)
  S = random.sample(L,card)
  T_in = []
  T_out = []
  for G in S:
    T_in.append(nx.laplacian_spectrum(G))
    T_out.append(nx.number_of_edges(G))
  T_in = np.array(T_in)
  T_out = np.array(T_out)
  return T_in, T_out

def np_torch_data(dataset):
  S = dataset.tensors[0]
  T = dataset.tensors[1]
  S_arr = S.numpy()
  T_arr = T.numpy()
  return S_arr, T_arr

def arr_data_eig(L,size):
  l = len(L)
  card = int(l*size)
  S = random.sample(L, card)
  T_in = []
  T_out = []
  for G in S:
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = torch.Tensor(A)
    v = G.number_of_nodes()
    C = torch.ones((v,1))
    D = torch.matmul(A, C)
    E = torch.matmul(A,D)
    F = torch.matmul(A, E)
    S = torch.reshape(torch.Tensor(np.real(nx.adjacency_spectrum(G))), (8,1))
    X = torch.cat([C,D,E,F,S], dim=1)
    X = X.numpy()
    T_in.append(X)
    T_out.append(nx.laplacian_spectrum(G))
  T_in = np.stack(T_in, axis=0)
  T_out = np.array(T_out)
  return T_in, T_out

def tensor_data_rad(array_dataset):
  T_in = []
  T_out = []
  arr_feats = array_dataset[0]
  arr_targets = array_dataset[1]
  v = np.shape(arr_feats)[1]
  aggregator = (1/v) * torch.ones((1,v))
  for i in range(len(arr_feats)):
    X = torch.Tensor(arr_feats[i])
    T_in.append(torch.matmul(aggregator, X))
    T_out.append(arr_targets[i][-1])
  T_in = torch.cat(T_in, dim=0)
  T_out = torch.Tensor(T_out)
  data = TensorDataset(T_in, T_out)
  return data

class Brouwer_predict(nn.Module):
  def __init__(self, num_nodes, input_dim, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_nodes = num_nodes
    self.num_feats = input_dim
    self.units = units
    self.lin1 = nn.Linear(self.num_feats,2*self.num_feats)
    self.conv1 = torch_geometric.nn.GINConv(nn.Sequential(self.lin1,nn.ReLU()))
    self.lin2 = nn.Linear(2*self.num_feats,4*self.num_feats)
    self.conv2 = torch_geometric.nn.GINConv(nn.Sequential(self.lin2,nn.ReLU()))
    self.lin3 = nn.Linear(4*self.num_feats,units)
    self.lin4 = nn.Linear(units,units)
    self.lin5 = nn.Linear(units,1)

  def forward(self, x, edge_tensor):
    out = self.conv1(x, edge_tensor)
    out = self.conv2(out, edge_tensor)
    out = self.lin3(out)
    out = nn.functional.relu(out)
    out = self.lin4(out)
    out = nn.functional.relu(out)
    out = self.lin5(out)
    return out

class EigToEdges(nn.Module):
  def __init__(self, input_dim, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_nodes = input_dim
    self.num_edges = int((self.num_nodes*(self.num_nodes-1))//2)
    self.lin1 = nn.Linear(self.num_nodes,units)
    self.lin2 = nn.Linear(units,units)
    self.lin3 = nn.Linear(units,units)
    self.lin4 = nn.Linear(units,units)
    self.lin5 = nn.Linear(units,self.num_edges)
    self.prob = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.lin1(x)
    out = nn.functional.relu(self.lin2(out))
    out = nn.functional.relu(self.lin3(out))
    out = nn.functional.relu(self.lin4(out))
    out = self.lin5(out)
    out = self.prob(out)
    return out

def pre_process(spec):
  V = len(spec)
  spec = torch.Tensor(spec)
  return torch.reshape(spec, (V,1))

class VectorRNN(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(VectorRNN, self).__init__()
    self.hidden_dim = hidden_dim
    self.rnn = nn.RNN(input_dim, hidden_dim)

  def forward(self, x):
    out = self.rnn(x)
    return out[-1]

class EdgePredictor(nn.Module):
  def __init__(self, num_nodes, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_nodes = num_nodes
    self.num_edges = int((self.num_nodes*(self.num_nodes-1))//2)
    self.vrnn = VectorRNN(1, self.num_edges)
    self.lin1 = nn.Linear(self.num_edges,self.num_edges)
    self.lin2 = nn.Linear(self.num_edges,self.num_edges)
    self.lin3 = nn.Linear(self.num_edges,self.num_edges)
    self.prob = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.vrnn(x)
    out = nn.functional.relu(self.lin1(out))
    out = nn.functional.relu(self.lin2(out))
    out = nn.functional.relu(self.lin3(out))
    out = self.prob(out)
    return out


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

num_nodes = 10
num_edges = num_nodes * (num_nodes - 1) // 2
num_episodes = 6 * (2 ** (num_edges +1) -1)
#num_episodes = 100
env = LinEnvMau(num_nodes)
# state = env.reset()
#print(env.state)

folder = "./graph_db/"
all_graphs = data_collector(folder, num_nodes, False)

print(len(all_graphs))
print(type(all_graphs[0]))

#wagner_scores = [(Graph(nx_graph).wagner1(), Graph(nx_graph).is_connected()) for nx_graph in all_graphs]

#print(wagner_scores)

counter_examples_indexes = []
for i, nx_graph in enumerate(all_graphs):
    if i % 1000 == 0:
      print(f"\rgraph {i}/{len(all_graphs)}", end="")
      sys.stdout.flush()

    graph = Graph(nx_graph)
    if graph.is_connected():
        if graph.wagner1() > 0:
            print(f"Condition does not hold for graph #{i}")
            counter_examples_indexes.append(i)
            # const = 1 + np.sqrt(graph.num_nodes - 1)
            # radius = max(np.real(nx.adjacency_spectrum(graph.graph)))
            # weight = len(nx.max_weight_matching(graph.graph))
            # wagner1_score = const - (radius + weight)

            # # Create a figure and axes
            # fig, ax = plt.subplots()
            # pos = nx.spring_layout(graph.graph)
            # ax.clear()
            # # Draw the new graph
            # plt.title(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")
            # nx.draw(graph.graph, pos=pos, ax=ax, with_labels=True)
            # # Update the display
            # plt.draw()
            # # Pause for a moment to show the plot
            # plt.pause(1)
            # # Keep the window open
            # #plt.show()
            # break
# else:
#     print("Condition holds for all connected graphs.")

print(counter_examples_indexes)

# print(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")

# graph = Graph(all_graphs[8])
# const = 1 + np.sqrt(graph.num_nodes - 1)
# radius = max(np.real(nx.adjacency_spectrum(graph.graph)))
# weight = len(nx.max_weight_matching(graph.graph))

# print(nx.adjacency_matrix(graph.graph).toarray())
# print(nx.adjacency_spectrum(graph.graph))

# print(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")
            
exit(0)

Q, policy, episodes, Q_diff_norms = mc_control_epsilon_greedy(env, num_episodes, discount_factor=0.1, epsilon=1, max_steps=1000, save=True)

print(episodes[:2])

states = [state for episode in episodes for state, _, _, _ in episode]
next_states = [next_state for episode in episodes for _, _, _, next_state in episode]
all = states + next_states # all states visited during episodes, including final states
all = [tuple(s) for s in all] # later we need it immutable

assert len(all) == num_episodes * num_edges * 2

all_unique = list(dict.fromkeys(all))

assert len(all_unique) == 2 ** (num_edges +1) -1

frequencies = []

for state in all_unique:
  frequencies.append(all.count(state) / len(all))

print(sum(frequencies))
print(1/len(frequencies), np.min(frequencies), np.max(frequencies))

from scipy.special import kl_div

def relative_entropy(p):
    # Calculate the uniform distribution
    uniform_p = np.ones(len(p)) / len(p)
    
    # Calculate the Kullback-Leibler divergence
    kl_divergence = kl_div(p, uniform_p)
    
    return np.sum(kl_divergence)


print(relative_entropy(frequencies))

import matplotlib.pyplot as plt

# Assume Q_diff_norms is the list of norms you got from your function

plt.figure(figsize=(10,6))
plt.plot(Q_diff_norms)
plt.title('Norm of Q-value Differences Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Norm of Q-value Differences')
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


wagner = nn.Sequential(
    nn.Linear(2*E, 2*E),
    nn.LeakyReLU(),
    nn.Linear(2*E,2*V),
    nn.ReLU(),
    nn.Linear(2*V,2*V),
    nn.ReLU(),
    nn.Linear(2*V,2*V),
    nn.ReLU(),
    nn.Linear(2*V,1),
    nn.Sigmoid())

summary(wagner,(2*E,))


a_train = player.one_shot_training(1024,20)

a_train

print(a_train[3])

player.env.miner.invariants['wagner-1'](a_train[3])

Env.score_traj

H = player.success_hist

[player.env.miner.invariants['wagner-1'](H[i]) for i in range(len(H))]

W = 18
G = int((W*(W-1))//2)
Env_s = LinEnv(W)

wagner_18 = nn.Sequential(
    nn.Linear(2*G, 2*W),
    nn.LeakyReLU(),
    nn.Linear(2*W,2*W),
    nn.ReLU(),
    nn.Linear(2*W, 2),
    nn.ReLU(),
    nn.Linear(2,1),
    nn.Sigmoid())

summary(wagner_18,(2*G,))

player = Agent(Env_s, wagner_18, l_r=0.001)

player.one_shot_training(300,25)

######################################

T = sess_gen(Env,wagner,1)

T

L = [1,2,3]
l = len(L)
L[l-1:]

#############################

best_cap = 32
buff_cap = 64

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(wagner.parameters(), lr=0.1)

num_eps = 16

buffer = []

for _ in range(1000):

  print(f"Iteration {_} starts")

  new_individuals = sess_gen(Env,wagner,80)
  best = best_individuals(new_individuals, best_cap)

  if best == []:
    continue

  print(f"Best current score is {best[0][2]}, obtained by")
  nx.draw(best[0][3], with_labels=True)
  plt.show()
  plt.close()

  best += buffer
  buffer = best_individuals(best, buff_cap)

  print(f"Best overal score is {buffer[0][2]}")

  inputs = []
  targets = []
  for i in range(len(buffer)):
    inputs += buffer[i][0]
    targets += buffer[i][1]

  inputs = torch.stack(inputs)
  targets = torch.Tensor(targets)

  for t in range(num_eps):
    prediction = wagner(inputs)
    prediction = prediction.squeeze()
    loss = criterion(prediction, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t == num_eps-1:
      print(f'Loss: {loss.item()}')

F = feats_miner(5)

S = sess_gen(Env,wagner,10)

g = best_individuals(S, 5)[0][3]

nx.adjacency_spectrum(g)

F.invariants["wagner-1"](g)

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
############

np.zeros(10)

T = LinEnv(5)
T.reset()
while not T.done:
  T.step(1)
T.history

####################

Exp = ExperimentData('wagner-1')

Exp.load_database('linear')

Exp.add_group('prova', meta=Exp.get_group_meta(E))

Exp.add_subgroup('prova', 'subprova', meta={'algorithm': 'MonteCarlo', 'policy': 'eps-greey'})

Exp.add_simulation('subprova', 'MC', (matrixes_per_episode, sar_per_episode), meta={'eps': 0.9})

Exp.data_path

Exp.groups_path

Exp.subgroups_dep

G = data_collector(path, 5, True)

inv = [fm.invariants['wagner-1'](g) for g in G]
for i in inv:
  print(i)

nx.adjacency_spectrum(G[0])


G = L[138]
torch.reshape(torch.Tensor(np.real(nx.adjacency_spectrum(G))), (8,1))


V = 8
E = int(((V*(V-1))//2))
units = 128

data_cc = arr_data_cc(L,1)

CcMLP = MLPClassifier(hidden_layer_sizes=(units, units, units), activation='relu', solver='sgd', learning_rate_init=0.001, momentum=0.9, max_iter=200, tol=1e-5, verbose=True)

data_ed = arr_data_ed(C,1)

EdgeMLP = MLPClassifier(hidden_layer_sizes=(units, units, units), activation='relu', solver='sgd', learning_rate_init=0.001, momentum=0.9, max_iter=1000, tol=1e-5, verbose=True)

data_eig = arr_data_eig(C,1)

tens_data_eig = tensor_data_rad(data_eig)

TT = random_split(tens_data_eig, [0.7, 0.3])

EigNet = nn.Sequential(nn.Linear(5,256), nn.ReLU(),
                       nn.Linear(256,256), nn.ReLU(),
                       nn.Linear(256,256), nn.ReLU(),
                       nn.Linear(256,64), nn.ReLU(),
                       nn.Linear(64,16), nn.ReLU(),
                       nn.Linear(16,1))
summary(EigNet, input_size=(1,5))

train = DataLoader(TT[0], batch_size=16, shuffle=True)
test = TT[1]

criterion = nn.L1Loss()
optimizer = optim.Adam(EigNet.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for inputs, targets in train:
        optimizer.zero_grad()
        outputs = EigNet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

EigNet.eval()
with torch.no_grad():
    inputs, targets = test[:]
    outputs = EigNet(inputs)
    mse = nn.functional.l1_loss(outputs, targets)
print(f"MSE: {mse}")

EigNet(tens_data_eig.tensors[0][5])

tens_data_eig.tensors[1][5]

EigMLP = MLPRegressor(hidden_layer_sizes=(units, 64, 1), activation='relu', solver='adam')

X_train, X_test, y_train, y_test = train_test_split(data_eig[0], data_eig[1], test_size=0.2, random_state=42)

skf = KFold(n_splits=5)

scores_cc = cross_val_score(CcMLP, data_cc[0], data_cc[1], cv=skf, scoring='accuracy')

scores_cc

scores_ed = cross_val_score(EdgeMLP, data_ed[0], data_ed[1], cv=skf, scoring='accuracy')

scores_ed


G = C[-1]
G
torch_graph = from_networkx(G)
eds = torch_graph.edge_index
eds

Net = Brouwer_predict(8, 2, 32)

X = torch.ones((8,1))
X

A = torch.ones((8,8))-torch.eye(8)
A

Y = torch.matmul(A,X)
Y

Z = torch.cat([X,Y], dim=1)
Z

Net(Z, eds)


EigClassifier = keras.Sequential()
EigClassifier.add(layers.Dense(units, activation=None))
EigClassifier.add(layers.Dense(units, activation='relu'))
EigClassifier.add(layers.Dense(units, activation='relu'))
EigClassifier.add(layers.Dense(units, activation='relu'))
EigClassifier.add(layers.Dense(E, activation='softmax'))
optim = optimizers.SGD(learning_rate=0.001, momentum=0.9)
EigClassifier.build((None,V))
EigClassifier.compile(loss="categorical_crossentropy", optimizer=optim)
EigClassifier.summary()

scores

Net = EigToEdges(8, 128)
summary(Net, (1,8))

dataset = tensor_data(C,1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(Net.parameters(), lr=0.01)

folds = skf.split()

num_epochs = 100
for epoch in range(num_epochs):
  print(f"Epoch {epoch} starts")
  train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
  for x, y in train_loader:
    optimizer.zero_grad()
    preds = Net(x)
    loss = criterion(preds, y)
    print(loss.item())
    loss.backward()
    optimizer.step()

nx.number_of_edges(L[56])

nx.draw(L[56])

x = torch.reshape(torch.Tensor(nx.laplacian_spectrum(L[56])), (1,8))
x

Net(x)


X = pre_process(nx.laplacian_spectrum(L[5]))


EP = EdgePredictor(4)

EP(X)