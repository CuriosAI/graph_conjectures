import copy
import numpy as np

import gymnasium as gym
from gymnasium import spaces
# To register the environment
from gym.envs.registration import register

import networkx as nx
import matplotlib.pyplot as plt

# my classes
#from feats_miner import feats_miner
from graph import Graph

class LinEnv(gym.Env):

  # Actions: remove or insert an edge in the graph
  REMOVE = 0
  INSERT = 1

  def __init__(self, number_of_nodes, normalize_reward):

    super(LinEnv, self).__init__()
    self.action_space = spaces.Discrete(2)
    assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
    self.number_of_nodes = number_of_nodes
    self.number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
    self.observation_space = spaces.MultiBinary(2 * self.number_of_edges)
    self.number_of_states = 2 ** self.number_of_edges - 1
    self.normalize_reward = normalize_reward # Normalize rewards when training, not when evaluating
    self.reset() # here self.state is created

  def step(self, action):

    # Extract the graph and timestep from the state
    graph = self.state[:self.number_of_edges]
    timestep = self.state[self.number_of_edges:]

    # Find the index of the edge to act upon
    edge_index = np.argmax(timestep)

    # Update the graph based on the action
    if action == self.REMOVE:
      graph[edge_index] = 0
    else:
      graph[edge_index] = 1

    # Update the timestep
    if edge_index < self.number_of_edges - 1:
      timestep[edge_index] = 0
      timestep[edge_index + 1] = 1
    else:
      timestep[edge_index] = 0 # The terminal state has timestep part all zero
      self.done = True

    # Combine the graph and timestep to form the new state of the env after the action
    self.state = np.concatenate([graph, timestep]) # From now on, self.state is the next state

    # Calculate the reward and set info dictionary
    if edge_index < self.number_of_edges - 1:
      info = {}
      reward = 0.0
    elif not Graph(graph).is_connected():
      #reward = float('-inf')
      reward = -1.0 if self.normalize_reward else -5.0 # penalty to be used when the conjecture holds only for connected graphs. this normalization assumes that other rewards are > -1
      #reward = 0.0
      # make_vec_env resets automatically when a done signal is encountered
      # we use info to pass the terminal state
      info = {}
      # info['terminal_state'] = copy.deepcopy(self.state) # not needed, because make_vec_env already does this
    else:
    # print(f"reward term = {Graph(graph).wagner1()}")
      # We normalize dividing by number_of_nodes, because empirically we see that min(wagner1())~-number_of_nodes. It should be proved.
      reward = Graph(graph).wagner1()/self.number_of_nodes if self.normalize_reward else Graph(graph).wagner1()
      # make_vec_env resets automatically when a done signal is encountered
      # we use info to pass the terminal state
      info = {}
      # info['terminal_state'] = copy.deepcopy(self.state) # not needed, because make_vec_env already does this

    # For gymnasium compatibility, step() must return a tuple of 5 elements:
    # state, reward, reward, terminated, truncated, info.
    return copy.deepcopy(self.state), reward, self.done, False, info

  def reset(self, seed=None, graph=None, options=None):

    """
    Reset the environment to a new initial state with the given graph part.

    Args:
        graph: The new initial graph part to be set (optional).
    """

    super().reset(seed=seed)
    timestep = np.zeros(self.number_of_edges, dtype=np.int8)
    timestep[0] = 1 # Starting state, next action will modify the first edge
    self.done = False # Episodes start with a non-terminal state by definition

    if graph is not None:
        # Create the full state by concatenating the graph with the initial timestep part
        self.state = np.concatenate((graph, timestep))
    else:
        # Set the graph part to the empty graph
        graph = np.zeros(self.number_of_edges, dtype=np.int8)
        self.state = np.concatenate((graph, timestep))

    return copy.deepcopy(self.state), {}

  def render(self):
    Graph(self.state[:self.number_of_edges]).draw()
    
# def register_linenv(number_of_nodes):
def register_linenv(number_of_nodes, normalize_reward):
  # id=f'LinEnv-{number_of_nodes}_nodes-normalize_{normalize_reward}'
  gym.register(
      # id=id,
      id='LinEnv-v0', # this name is hard-coded in rl_zoo3/hyperparams/ppo.yml, we cannot change it
      entry_point='envs:LinEnv',
      kwargs={'number_of_nodes': number_of_nodes, 'normalize_reward': normalize_reward}
  )
  # return id
