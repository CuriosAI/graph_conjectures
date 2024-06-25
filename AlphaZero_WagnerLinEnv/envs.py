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
  REMOVE = 0  # or DON'T ADD
  INSERT = 1

  REWARDS = ['wagner', 'brouwer']

  def __init__(self, number_of_nodes, reward, normalize_reward=True):

    super(LinEnv, self).__init__()
    self.number_of_nodes = number_of_nodes
    self.number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
    # Originals
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.MultiBinary(2 * self.number_of_edges)

    # Action & Observation spaces (Giulia)
    # self.action_space = spaces.Discrete(number_of_nodes * (number_of_nodes - 1))
    # self.observation_space = spaces.Box(low=0, high=1, shape=(number_of_nodes, number_of_nodes), dtype=np.float32)

    assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"

    self.action_size = 2 * self.number_of_edges
    self.column_count = number_of_nodes
    self.row_count = number_of_nodes
    # self.number_of_states = 2 ** self.number_of_edges - 1

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

  def score(self):  # c'è una reference in AlphaZeroParallel?
    timestep = self.state[self.number_of_edges:]
    edge_index = np.argmax(timestep)

    if self.reward == 'wagner':
      if edge_index < self.number_of_edges - 1:
        info = {}
        reward = 0.0

      elif not Graph(self.state, True).is_connected():
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
        reward = Graph(self.state, True).wagner1() / self.number_of_nodes if self.normalize_reward else Graph(
          self.state, True).wagner1()
        info = {'connected'}
      return reward, info

    if self.reward == 'brouwer':
      # aggiungere, x ora ritorna 0
      reward = Graph(self.state, True).brouwer()
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
      graph = np.zeros(self.number_of_edges, dtype=np.int8)
      self.state = np.concatenate((graph, timestep))

    return copy.deepcopy(self.state)

  def render(self):
    Graph(self.state, True).draw()

  # ALPHAZERO SUITABILITY
  # ------------------------------------------------------------------
  def get_valid_moves(self):
    return self.state[self.number_of_edges:]

  # Non va bene per la versione parallela
  def get_encoded_state(self):
    adj = Graph(self.state, False).graph  # 1 matrice sola
    if isinstance(adj, np.ndarray):
      print('True')
    encoded_state = np.stack((adj == 0, adj == 1)).astype(np.float32)
    # This is needed in case of parallel execution of Alphazero
    if len(adj.shape) == 3:
      encoded_state = np.swapaxes(encoded_state, 0, 1)

    return encoded_state

  # Al momento mi sembra necessaria.
  # Non vedo come tradurla in modo più game indip
  # Miglioriamo leggermente l'interazione tra check_win e get_value_and_terminated
  # non voglio calcolare lo score due volte
  # il calcolo dello spettro del laplaciano potrebbe essere oneroso!

  def check_win(self):
    win = False
    reward = 0
    info = {'disconnected'}
    if Graph(self.state, True).is_connected():
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

  ''' Vecchi commenti '''
  #   # -------- Flora --------
  #   # Surrogato di get_initial_state()
  #   # modificare con azione, se serve
  #   # def get_state(self, action):
  #   #    return Graph(self.state,False)
  #   #   # ma serviva davvero? Come sono arrivata quì?
  #   #   # Da Wagnerplay : get_initial_state ritorna la matrice triangolare sup
  #
  #   def get_encoded_state(self, state):
  #     # in LinEnv, state è un vettore. Bisogna ricostruire la matrice
  #     graph = state[:self.number_of_edges]
  #     A = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.uint8)
  #
  #     edge_index = 0
  #     for i in range(number_of_nodes):
  #       for j in range(i + 1, number_of_nodes):
  #         if graph[edge_index] == 1:
  #           A[i, j] = 1
  #         edge_index += 1
  #
  #     encoded_state = np.stack((A == 0, A == 1)).astype(np.float32)
  #     # This is needed in case of parallel execution of Alphazero
  #     if len(A.shape) == 3:
  #       encoded_state = np.swapaxes(encoded_state, 0, 1)
  #
  #     return encoded_state
  #
  #   def get_next_state(self):
  #     # il metodo step dovrebbe aggiornare il gioco
  #     # questo get_next_state serve solo per i self-play games
  #     A = self.get_state()
  #     next = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.uint8)
  #     j, k = self.next_edge()
  #     next[j, k] = 1
  #     return A + next
  #     # ritorna la matrice che si ottiene se aggiungo il nuovo arco,
  #     # ma serviva davvero? Come sono arrivata quì?
  #     # Da Wagnerplay : get_initial_state ritorna la matrice triangolare sup
  #
  #   def get_valid_moves(self):
  #     # recover current position from state
  #     # compute next edge (i,j) in our (fixed) ordering
  #     # available actions are:
  #     # add (i,j)
  #     # don't add (i,j)
  #     valid_moves = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.uint8)
  #     j, k = self.next_edge()
  #     valid_moves[j, k] = 1
  #
  #     return valid_moves.flatten()
  #     # In questa versione, l'azione opposta dovrebbe essere calcolata con get_encoded_states
  #
  # # ----------------------------------

  '''Nella versione di Giulia, le mosse possibili sono sempre rappresentate da una matrice
    valid_moves[i,j] = 1 sse togliere l'arco (i,j) non sconnette il grado
    Manteniamo la rappresentazione matriciale, ma valid_moves[i,j] = 1 indicherà l'aggiunta dell'arco
    In sintesi: l'azione è rappresentata dal grafo che si ottiene modificando 
    (PERO': C'è anche l'azione di lasciare tutto com'è'''
    # get_valid_moves Giulia
    # n = len(state)
    # valid_moves = np.zeros((n, n), dtype=np.uint8)
    # # Assuming a symmetric graph, you only need to check one triangle
    # triu_indices = np.triu_indices(n, k=1)
    # for i, j in zip(*triu_indices):
    #   # Temporarily modify state
    #   original_value = state[i, j]
    #   state[i, j] = 0  # Example modification, adapt based on actual logic needed
    #   if self.check_connectivity(state):
    #     valid_moves[i, j] = 1
    #   # Restore original state
    #   state[i, j] = original_value
    # return valid_moves.flatten()
    ''' Se non ho capito male: viene restituita una matrice in cui gli uni
    corrispondono agli archi che è possibile togliere
    senza sconnettere il grafo?'''


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
