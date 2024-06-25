import copy
import numpy as np

# import gymnasium as gym
# from gymnasium import spaces
import gymnasium as gym
from gymnasium import spaces

# To register the environment
from gymnasium.envs.registration import register

import networkx as nx
import matplotlib.pyplot as plt

# my classes
from graph import Graph


class LinEnv(gym.Env):
  # Actions: remove or insert an edge in the graph
  REMOVE = 0  # or DON'T ADD
  INSERT = 1

  REWARDS = ['wagner', 'brouwer']

  def __init__(self, number_of_nodes, reward, normalize_reward=True):

    super(LinEnv, self).__init__()
    self.number_of_nodes = number_of_nodes
    self.number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
    # Originals
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.MultiBinary(2 * self.number_of_edges) # ok x GNN

    # Action & Observation spaces (Giulia)
    # self.action_space = spaces.Discrete(number_of_nodes * (number_of_nodes - 1))
    # self.observation_space = spaces.Box(low=0, high=1, shape=(number_of_nodes, number_of_nodes), dtype=np.float32)

    assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"

    self.action_size = 2 * self.number_of_edges
    self.column_count = number_of_nodes
    self.row_count = number_of_nodes

    self.state = None
    self.player = None  # <-- x tenere traccia delle azioni nel MCTS
    self.done = False  # <-- inutilizzato, x ora

    # Reward
    assert reward in self.REWARDS, "unavailable reward"
    self.reward = reward
    self.normalize_reward = normalize_reward
    # Normalize rewards when training, not when evaluating
    # Va messo a False durante il gioco

    self.reset()  # here self.state is created

  def reset(self, seed=None, graph=None, options=None):  # settaggio del seed negli esperimenti?
    """
    Reset the environment to a new initial state with the given graph part.
    Args:
        graph: The new initial graph part to be set (optional).
    """

    super().reset(seed=seed)
    timestep = np.zeros(self.number_of_edges, dtype=np.int8)

    timestep[0] = 1  # Starting state, next action will modify the first edge
    self.done = False  # Episodes start with a non-terminal state by definition

    if graph is not None:
      # Create the full state by concatenating the graph with the initial timestep part
      self.state = np.concatenate((graph, timestep))
    else:
      # Set the graph part to the empty graph
      #graph = np.zeros(self.number_of_edges, dtype=np.int8)
      graph = np.ones(self.number_of_edges, dtype=np.int8)
      self.state = np.concatenate((graph, timestep))

    return copy.deepcopy(self.state)

  def score(self):  # c'è una reference in AlphaZeroParallel?
    timestep = self.state[self.number_of_edges:]
    edge_index = np.argmax(timestep)

    if self.reward == 'wagner':
      if edge_index < self.number_of_edges - 1:
        info = {}
        reward = 0.0

      elif not Graph(self.state, nx_g=True).is_connected():
        reward = -1.0 if self.normalize_reward else -5.0  # penalty to be used when the conjecture
        # holds only for connected graphs.
        # This normalization assumes that other rewards are > -1
        # we use info to pass the terminal state
        # (Flora) : info può essere utilizzato x mantenere l'informazione sulla connessione
        # in modo da non doverla ricontrollare (modifica in sospseso)
        info = {'not_connected'}

      else:
        # print(f"reward term = {Graph(graph).wagner1()}")
        # We normalize dividing by number_of_nodes,
        # because empirically we see that min(wagner1())~-number_of_nodes.
        # It should be proved.
        reward = Graph(self.state, nx_g=True).wagner1() / self.number_of_nodes if self.normalize_reward else Graph(
          self.state, nx_g=True).wagner1()
        info = {'connected'}
      return reward, info

    if self.reward == 'brouwer':
      # aggiungere, x ora ritorna 0
      reward = Graph(self.state, nx_g=True).brouwer()
      info = {}
      return reward, info

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
      timestep[edge_index] = 0  # The terminal state has timestep part all zero
      self.done = True  # legato alla funzione check_win e get_value_and_terminated

    # Combine the graph and timestep to form the new state of the env after the action
    self.state = np.concatenate([graph, timestep])  # From now on, self.state is the next state

    # --- Generalized reward ---
    # Calculate the reward and set info dictionary
    reward, info = self.score()

    # For gymnasium compatibility, step() must return a tuple of 5 elements:
    # state, reward, reward, terminated, truncated, info.
    return copy.deepcopy(self.state), reward, self.done, False, info


  def render(self):
    Graph(self.state, nx_g=True).draw()

  # ALPHAZERO SUITABILITY
  # ------------------------------------------------------------------
  def get_valid_moves(self):
    return self.state[self.number_of_edges:]

  # Non va bene per la versione parallela
  def get_encoded_state(self):
    adj = Graph(self.state).graph  # 1 matrice sola
    # if isinstance(adj, np.ndarray):
    #   print('True')
    # encoded_state = np.stack((adj == 0, adj == 1)).astype(np.float32)

    return adj

  # Al momento mi sembra necessaria.
  # Non vedo come tradurla in modo più game indip
  # Miglioriamo leggermente l'interazione tra check_win e get_value_and_terminated
  # non voglio calcolare lo score due volte
  # il calcolo dello spettro del laplaciano potrebbe essere oneroso!

  def check_win(self):
    win = False
    reward = 0
    info = {'disconnected'}
    if Graph(self.state, nx_g=True).is_connected():
      reward, info = self.score()  # info = {'connected'}
      if reward > 1e-12:
        win = True
    return win, reward, info

  # def end_game(self):
  #   return np.sum(self.state[self.number_of_edges:]) == 0

  def get_value_and_terminated(self):
    value = 0.00
    reward, win, info = self.check_win()
    if win:
      value = reward / self.number_of_nodes if self.normalize_reward else reward
      return value, True
      # Abbiamo vinto: comprende due casi
      # 1) Configurazione vincente, partita finita
      # 2) Configurazione vincente, partita non finita
    else:
      # Potremmo comunque aver terminato la partita
      if np.sum(self.state[self.number_of_edges:]):
        terminated = False
      else:
        terminated = True
      if 'connected' in info:
        value = 0
        return value, terminated
      else:
        value = -1 if self.normalize_reward else -5
        return value, terminated

  def get_reward(self):
    return self.get_value_and_terminated()[0]  # utilizzato??

  def extract_action(self, k):
    n = self.number_of_nodes
    # valore di controllo errori
    if k >= n * (n - 1):  # questo caso non dovrebbe verificarsi mai
      return -1
    elif k < n * (n - 1) / 2:
      return 1
    else:
      return 0

  # def register_linenv(number_of_nodes):
  def register_linenv(number_of_nodes, normalize_reward):
    # id=f'LinEnv-{number_of_nodes}_nodes-normalize_{normalize_reward}'
    gym.register(
      # id=id,
      id='LinEnv-v0',  # this name is hard-coded in rl_zoo3/hyperparams/ppo.yml, we cannot change it
      entry_point='envs:LinEnv',
      kwargs={'number_of_nodes': number_of_nodes, 'normalize_reward': normalize_reward}
    )