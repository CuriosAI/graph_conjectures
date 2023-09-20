import copy
import numpy as np

import gymnasium as gym
from gymnasium import spaces
# To register the environment
from gym.envs.registration import register

import networkx as nx
import matplotlib.pyplot as plt

# my classes
from graph import Graph

class LinEnv(gym.Env):

  # Actions: remove or insert an edge in the graph
  REMOVE = 0
  INSERT = 1

  def __init__(self, number_of_nodes, normalize_reward=True):

    super(LinEnv, self).__init__()
    self.action_space = spaces.Discrete(2)
    self.number_of_actions = self.action_space.n
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
    # print(f'graph after action {action} is {graph}')
    # input()

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
    elif not Graph(self.state).is_connected():
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
      reward = Graph(self.state).wagner1()/self.number_of_nodes if self.normalize_reward else Graph(self.state).wagner1()
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
    Graph(self.state).draw()
    
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

####################
# CHATGPT PROPOSAL
####################

class LocEnv(gym.Env):

  # Define the possible actions: change (1) or not change (0) an edge in the graph
  CHANGE = 1
  NOT_CHANGE = 0

  def __init__(self, number_of_nodes, max_steps, normalize_reward=True):

    super(LocEnv, self).__init__()
    
    # Define the action space: choose a node and decide to change or not change the edge
    self.action_space = spaces.Tuple((spaces.Discrete(number_of_nodes), spaces.Discrete(2)))
    self.number_of_actions = 2 * number_of_nodes

    # Initialize environment variables
    self.number_of_nodes = number_of_nodes
    self.number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2 # Not needed
    self.max_steps = max_steps
    
    # Define the observation space: includes both the adjacency matrix and the current node
    self.observation_space = spaces.Dict({
        'adjacency_matrix': spaces.Box(low=0, high=1, shape=(number_of_nodes, number_of_nodes), dtype=np.int8),
        'current_node': spaces.Discrete(number_of_nodes)
    })
    
    self.current_node = 0  # Start at node 0
    self.current_step = 0  # Initialize step counter
    self.normalize_reward = normalize_reward  # Whether to normalize rewards
    self.done = False  # Initialize 'done' flag
    self.reset() # Here self.state is created

  def step(self, action):

    # Extract the target node and the decision to change or not change the edge from the action
    target_node, change_edge = action

    # Update the adjacency matrix based on the action
    if change_edge == self.CHANGE:
      # XOR operation to flip the bit, effectively changing the edge
      self.adjacency_matrix[self.current_node, target_node] ^= 1
      self.adjacency_matrix[target_node, self.current_node] ^= 1

    # Update the current node to the target node
    self.current_node = target_node

    # From now on, self.state is the next state
    self.state = {'adjacency_matrix': self.adjacency_matrix, 'current_node': self.current_node}

    # Increment the step counter
    self.current_step += 1

    # Check if the episode should end
    if self.current_step >= self.max_steps:
      self.done = True

    # Calculate the reward based on the final graph
    reward = 0.0
    info = {}
    if self.done:
      # Calculate reward based on graph connectivity and the 'wagner1' score
      if not Graph(self.state).is_connected():
        reward = -1.0 if self.normalize_reward else -5.0
        info = {}
      else:
        reward = Graph(self.state).wagner1() / self.number_of_nodes if self.normalize_reward else Graph(self.state).wagner1()
        info = {}

    # For gymnasium compatibility, step() must return a tuple of 5 elements:
    # state, reward, reward, terminated, truncated, info.
    return copy.deepcopy(self.state), reward, self.done, False, info

  def reset(self):

    """
    Reset the environment to a new initial state.
    """

    # Initialize the adjacency matrix to zeros
    self.adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.int8)
    
    # Reset current node and step counter
    self.current_node = 0
    self.current_step = 0
    
    # Reset 'done' flag
    self.done = False

    # Compile the state dictionary
    self.state = {'adjacency_matrix': self.adjacency_matrix, 'current_node': self.current_node}

    return copy.deepcopy(self.state), {}

  def render(self):
    # Assuming Graph(self.state[:self.number_of_edges]).draw() would draw the graph,
    # here it would be:
    Graph(self.state).draw()
    
# You'll need to implement the Graph class and its methods like is_connected() and wagner1()
