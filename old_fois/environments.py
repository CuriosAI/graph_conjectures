# create a custom environment for the graph game
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt

class GraphEdgeFlipEnvironment(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
  def __init__(self, render_mode=None, size=5):
    self.size = size
    self._n_different_edges = int(self.size * (self.size-1)/2)
    self.observation_space = spaces.Box(low=0, high=1, shape=(self._n_different_edges,), dtype=int)
    self.action_space = spaces.Discrete(self._n_different_edges);
    self._action_to_edge = {}

    idx = 0
    for i in range(self.size):
      for j in range(i):
        self._action_to_edge[idx] = np.array([i,j])
        idx = idx + 1

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
  def _get_obs(self):
    return np.copy(self._current_ints)

  def _get_info(self):
    return {} # TODO: implement this

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._nsteps = 0
    self._current_graph = nx.Graph()
    
    self._current_ints = np.ones((self._n_different_edges,))
    self._current_graph.add_nodes_from(range(self.size))
    idx = 0
    for i in range(self.size):
      for j in range(i):
        self._current_graph.add_edge(i,j)
        idx = idx + 1

    self.compute_value()
    self._start_value = self._value

    observation = self._get_obs()
    info = self._get_info()
    
    if self.render_mode == "human":
        self._render_frame()
        
    return observation, info

  def value(self):
    return self._value

  def compute_value(self):
    mu = len(nx.max_weight_matching(self._current_graph, maxcardinality=True))
    e = np.abs(np.real(nx.adjacency_spectrum(self._current_graph)))
    lambda_1 = max(e)
    self._value = (lambda_1 + mu)
    return self._value

  def step(self, action):
    edge = self._action_to_edge[action]
    terminated = False
    reward = 0

    if self._current_ints[action] == 1:
      self._current_ints[action] = 0
      self._current_graph.remove_edge(edge[0],edge[1])
      if not(nx.is_connected(self._current_graph)):
        self._current_ints[action] = 1
        self._current_graph.add_edge(edge[0],edge[1])
    # else:
    #   self._current_ints[action] = 1
    #   self._current_graph.add_edge(edge[0],edge[1])

    observation = self._get_obs()
    info = self._get_info()
   
    self.compute_value()

    self._nsteps = self._nsteps + 1
    if self._nsteps >= 1.3*(self.size * (self.size - 1)/2 - self.size + 1) or len(self._current_graph.edges) == self.size - 1:
      terminated = True
    
    if self._value < np.sqrt(self.size -1 ) + 1:
      terminated = True
      print(self._current_ints)
      self._render_frame()
      print(nx.adjacency_matrix(self._current_graph))
    
    if terminated:
      reward += self._start_value - self._value 
      
    if self.render_mode == "human":
        self._render_frame()
        
    return observation, reward, terminated, False, info

  def draw(self):
    nx.draw(self._current_graph)
    
  def render(self):
    if self.render_mode == "rgb_array":
        return self._render_frame()
      
  def _render_frame(self):
    plt.figure(1)
    plt.clf()
    self.draw()
    plt.pause(0.001)
    
    
    
    
    
class SecondNeighborhoodEnvironment(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
  def __init__(self, render_mode=None, size=5):
    self.size = size
    self._n_different_edges = int(self.size * (self.size-1)/2)
    self.observation_space = spaces.Box(low=0, high=1, shape=(self._n_different_edges,), dtype=int)
    self.action_space = spaces.Discrete(self._n_different_edges);
    self._action_to_edge = {}

    idx = 0
    for i in range(self.size):
      for j in range(i):
        self._action_to_edge[idx] = np.array([i,j])
        idx = idx + 1

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
  def _get_obs(self):
    return np.copy(self._current_ints)

  def _get_info(self):
    return {} # TODO: implement this

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._nsteps = 0
    self._current_graph = nx.Graph()
    
    self._current_ints = np.ones((self._n_different_edges,))
    self._current_graph.add_nodes_from(range(self.size))
    idx = 0
    for i in range(self.size):
      for j in range(i):
        self._current_graph.add_edge(i,j)
        idx = idx + 1

    self.compute_value()
    self._start_value = self._value

    observation = self._get_obs()
    info = self._get_info()
    
    if self.render_mode == "human":
        self._render_frame()
        
    return observation, info

  def value(self):
    return self._value

  def compute_value(self):
    mu = len(nx.max_weight_matching(self._current_graph, maxcardinality=True))
    L = nx.adjacency_matrix(self._current_graph)
    e = np.abs(np.real(np.linalg.eigvals(L.toarray())))
    lambda_1 = max(e)
    self._value = (lambda_1 + mu)
    return self._value

  def step(self, action):
    edge = self._action_to_edge[action]
    terminated = False
    reward = 0
    old_value = self._value
    if self._current_ints[action] == 1:
      self._current_ints[action] = 0
      self._current_graph.remove_edge(edge[0],edge[1])
      if not(nx.is_connected(self._current_graph)):
        self._current_ints[action] = 1
        self._current_graph.add_edge(edge[0],edge[1])      

    observation = self._get_obs()
    info = self._get_info()
    self.compute_value()

    self._nsteps = self._nsteps + 1
    if self._nsteps >= (self.size * (self.size - 1)/2 - self.size + 1)*2:
      terminated = True
      
    
    if self._value < np.sqrt(self.size -1 ) + 1:
      terminated = True
      print(self._current_ints)
      self._render_frame()
      print(nx.adjacency_matrix(self._current_graph))
    
    if terminated:
      reward += old_value - self._value 
      
    if self.render_mode == "human":
        self._render_frame()
        
    return observation, reward, terminated, False, info

  def draw(self):
    nx.draw(self._current_graph)
    
  def render(self):
    if self.render_mode == "rgb_array":
        return self._render_frame()
      
  def _render_frame(self):
    plt.figure(1)
    plt.clf()
    self.draw()
    plt.pause(0.001)

