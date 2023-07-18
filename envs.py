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

class LinEnvMau(gym.Env):

  # Actions: remove or insert an edge in the graph
  REMOVE = 0
  INSERT = 1

  def __init__(self, number_of_nodes, normalize_reward):

    super(LinEnvMau, self).__init__()
    self.action_space = spaces.Discrete(2)
    assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
    self.number_of_nodes = number_of_nodes
    self.number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
    self.observation_space = spaces.MultiBinary(2 * self.number_of_edges)
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
    
# Register the environment
# from gym.envs.registration import register

# register(
#     id='LinEnvMau-v0',
#     #entry_point=__name__ + ':LinEnvMau',
#     entry_point='envs:LinEnvMau',
# )

# def register_linenv(number_of_nodes):
def register_linenv(number_of_nodes, normalize_reward):
  # id=f'LinEnvMau-{number_of_nodes}_nodes-normalize_{normalize_reward}'
  gym.register(
      # id=id,
      id='LinEnvMau-v0', # this name is hard-coded in rl_zoo3/hyperparams/ppo.yml, we cannot change it
      entry_point='envs:LinEnvMau',
      kwargs={'number_of_nodes': number_of_nodes, 'normalize_reward': normalize_reward}
  )
  # return id

# class LinEnv(gym.Env):
#     """
#     LinEnv is a custom environment class that simulates a graph with no loops with a
#     specified number of nodes. The class is designed to be compatible with OpenAI's
#     gym library.
#     Graphs are represented 'linearly' by their edges: [1,0,0,1,0,1] is a graph with 4 nodes (and thus at most 6 edges), where edges 0, 3 and 5 are present. Edges are ordered accordingly to node order: thus, [1,0,0,1,0,1] means that nodes (0,1), (1,2) and (2,3) are connected.
#     Given a graph and a timestep i, the agent can choose between 2 actions: insert the edge number i, or remove it. Thus, a state must contain also the timestep. A state is then an np array of lenght 2*edges, where the first edges elements represent the graph, and the last edges elements is a one-hot vector representing the timestep. For example, [1,0,0,1,0,1,0,0,1,0,0,0] means that edge number 2=(0,3) (starting from edge number 0) is the next edge we will act on.
#     If rew_types is 'sparse', the reward is the score we want to maximize, given only at the end of the episode.

#     Attributes:
#         name (str): Name of the environment ('linear').
#         rew_types (set): Set of reward types supported by the environment.
#         rew_type (str): The reward type used in the environment.
#         score (str): The scoring function used to evaluate the graph.
#         num_nodes (int): The number of nodes in the graph.
#         conn (bool): Whether the graph should be connected.
#         miner (object): An object for mining features from the graph.
#         score_ev (function): The scoring evaluation function.
#         obs_dim (int): The dimensionality of the observation space.
#         num_edges (int): The number of edges in the graph.
#         pointer (int): A pointer to the current position in the observation space.
#         observation_space (gym.spaces.MultiBinary): The observation space.
#         initial_state (np.array): The initial state of the environment.
#         state (np.array): The current state of the environment.
#         action_space (gym.spaces.Discrete): The action space.
#         reward_range (tuple): The range of possible rewards.
#         initial_adj (np.array): The initial adjacency matrix of the graph.
#         adj (np.array): The current adjacency matrix of the graph.
#         initial_ST (np.array): The initial state transition matrix.
#         ST (np.array): The current state transition matrix.
#         initial_graph (nx.Graph): The initial graph.
#         graph (nx.Graph): The current graph.
#         initial_scores (dict): The initial scores for each reward type.
#         initial_score (float): The initial score of the graph.
#         score (float): The current score of the graph.
#         score_traj (list): A list of scores over time.
#         done (bool): Whether the episode is done.
#         history (list): A list of historical states, actions, rewards, etc.

#     Methods:
#         time_up(i, j): Updates and returns the indices i, j.
#         step(action): Takes an action and returns the next state, reward, and whether the episode is done.
#         render(): Renders the current state of the graph.
#         obs_conversion(): Converts the current state to an observation.
#         reset(): Resets the environment to its initial state.
#     """

#     def __init__(self, num_nodes, score='wagner-1', rew_type='sparse', conn=True):
#         super(LinEnv, self).__init__()
#         self.name = 'linear'
#         self.rew_types = {'sparse','sparse_max','cont','cont_var'}
#         self.rew_type = rew_type
#         self.score_fun = score
#         self.num_nodes = num_nodes
#         self.conn = conn
#         self.miner = feats_miner(self.num_nodes)
#         self.score_ev = self.miner.invariants[self.score_fun]
#         self.obs_dim = self.num_nodes * (self.num_nodes-1)
#         self.num_edges = int(self.obs_dim//2)
#         self.pointer = copy.deepcopy(self.num_edges)
#         self.observation_space = gym.spaces.MultiBinary(self.obs_dim)
#         self.initial_state = np.zeros(self.obs_dim)
#         self.initial_state[self.num_edges] += 1
#         self.state = copy.deepcopy(self.initial_state)
#         self.action_space = gym.spaces.Discrete(2)
#         self.reward_range = (-float('inf'), float('inf'))
#         self.initial_adj = np.zeros((self.num_nodes,self.num_nodes))
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.initial_ST = np.zeros((self.num_nodes,2))
#         self.initial_ST[0,0] += 1
#         self.initial_ST[1,1] += 1
#         self.ST = copy.deepcopy(self.initial_ST)
#         self.initial_graph = nx.Graph(self.initial_adj)
#         self.graph = copy.deepcopy(self.initial_graph)
#         self.initial_scores = {'sparse': 0, 'sparse_max': 0, 'cont': self.score_ev(self.graph), 'cont_var': self.score_ev(self.graph)}
#         self.initial_score = copy.deepcopy(self.initial_scores[rew_type])
#         self.score = copy.deepcopy(self.initial_score)
#         self.score_traj = [copy.deepcopy(self.initial_score)]
#         self.done = False
#         self.history = []

#     def time_up(self, i, j):
#       if j == self.num_nodes-1:
#         i += 1
#         j = i + 1
#       else:
#         i = i
#         j += 1
#       return i, j

#     def step(self, action):
#         sarsd = [0,0,0,0,0]
#         h = self.obs_conversion()
#         sarsd[0] = h
#         sarsd[1] = action
#         st = np.nonzero(self.ST)[0].tolist()
#         i, j = st[0], st[1]
#         t = self.pointer
#         index = np.zeros(self.obs_dim)
#         index[t] -= 1
#         self.state += index
#         self.ST[i,0] -= 1
#         self.ST[j,1] -= 1
#         done = copy.deepcopy(self.done)
#         reward = 0
#         if self.pointer == self.obs_dim-1:
#           done = True
#           self.done = done
#         if action == 1:
#           self.state[self.pointer-self.num_edges] += 1
#           self.adj[i,j] += 1
#           self.adj[j,i] += 1
#           self.graph = nx.Graph(self.adj)
#           self.score = self.score_ev(self.graph)
#         self.score_traj.append(copy.deepcopy(self.score))
#         if not done:
#           self.pointer += 1
#           self.state[self.pointer] += 1
#           i, j = self.time_up(i,j)
#           self.ST[i,0] += 1
#           self.ST[j,1] += 1
#         next_state = copy.deepcopy(self.state)
#         if self.rew_type == 'sparse':
#           if done:
#             if self.conn == True:
#               if self.miner.invariants['num_cc'](self.graph) != 1:
#                 reward = -1000000.0
#               else:
#                 reward = self.score
#             else:
#               reward = self.score
#         if self.rew_type == 'sparse_max':
#           if done:
#             reward = max(self.score_traj)
#         if self.rew_type == 'cont':
#           reward = self.score
#         if self.rew_type == 'cont_var':
#           reward = self.score_traj[-1] - self.score_traj[-2]
#         sarsd[2] = reward
#         h_up = self.obs_conversion()
#         sarsd[3] = h_up
#         sarsd[4] = int(done)
#         self.history.append(sarsd)
#         return next_state, reward, done

#     def render(self):
#       nx.draw(self.graph, with_labels=True)
#       plt.show()
#       plt.close()

#     def obs_conversion(self):
#       e = self.num_edges
#       ist = copy.deepcopy(self.state)
#       st = copy.deepcopy(self.ST)
#       st = np.nonzero(st)[0].tolist()
#       id = int(np.dot(ist[:e], 2**np.arange(e)))
#       if st == []:
#         st = [0,0]
#         return (id, 0, 0)
#       else:
#         return (id, st[0], st[1])

#     def reset(self):
#         self.state = copy.deepcopy(self.initial_state)
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.ST = copy.deepcopy(self.initial_ST)
#         self.score = copy.deepcopy(self.initial_score)
#         self.pointer = copy.deepcopy(self.num_edges)
#         self.score_traj = [copy.deepcopy(self.initial_score)]
#         self.done = False
#         self.history = []
#         return self.initial_state, self.initial_score, self.done

# class LocEnv(gym.Env):
#     def __init__(self, num_nodes, score='wagner-1', rew_type='sparse', stop_type='fixed', chaos=0):
#         super(LocEnv, self).__init__()
#         self.name = 'local'
#         self.rew_types = {'sparse','sparse_max','cont','cont_var'}
#         self.rew_type = rew_type
#         self.score_fun = score
#         self.num_nodes = num_nodes
#         self.chaos = chaos
#         self.miner = feats_miner(self.num_nodes)
#         self.score_ev = self.miner.invariants[self.score_fun]
#         self.num_edges = int((self.num_nodes * (self.num_nodes-1))//2)
#         self.obs_dim = self.num_edges + self.num_nodes
#         self.observation_space = gym.spaces.MultiBinary(self.obs_dim)
#         self.initial_state = np.zeros(self.obs_dim)
#         self.initial_state[self.num_edges] += 1
#         self.state = copy.deepcopy(self.initial_state)
#         self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.num_nodes),gym.spaces.Discrete(2)))
#         self.reward_range = (-float('inf'), float('inf'))
#         self.initial_adj = np.zeros((self.num_nodes,self.num_nodes))
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.pointer = 0
#         self.initial_graph = nx.Graph(self.initial_adj)
#         self.graph = copy.deepcopy(self.initial_graph)
#         self.initial_scores = {'sparse': 0, 'sparse_max': 0, 'cont': self.score_ev(self.graph), 'cont_var': self.score_ev(self.graph)}
#         self.initial_score = copy.deepcopy(self.initial_scores[rew_type])
#         self.score = copy.deepcopy(self.initial_score)
#         self.score_traj = [copy.deepcopy(self.initial_score)]
#         self.time = 0
#         self.done = False
#         self.target_score = copy.deepcopy(self.initial_score)
#         self.stop_types = {'fixed': 1, 'dynamic': copy.deepcopy(self.target_score), 'pass': 1}
#         self.stop_type = stop_type
#         self.stop_param = copy.deepcopy(self.stop_types[self.stop_type])
#         if self.stop_type == 'pass':
#           self.waiting = 0
#         self.history = []

#     def triangulate(self, i,j):
#       i, j = i, j
#       S = np.ones((self.num_nodes,self.num_nodes), dtype=np.int32)
#       S = np.triu(S, k=1)
#       s = 0
#       kron = np.zeros(self.obs_dim)
#       if i == j:
#         return s, kron
#       if i > j:
#         h = j
#         j = i
#         i = h
#       for p in range(j):
#         for r in range(self.num_nodes):
#           s += S[r,p]
#       for q in range(i+1):
#         s += S[i,j]
#       return s, kron
#       kron[s-1] += 1
#       return s, kron

#     def stop_check(self):
#       stop = False
#       if self.stop_type == 'fixed':
#         if self.time == self.stop_param * self.num_edges:
#           stop = True
#       elif self.stop_type == 'dynamic':
#         if self.score_traj[-1] > self.stop_param:
#           stop = True
#       elif self.stop_type == 'pass':
#         if self.waiting == self.stop_param:
#           stop = True
#       return stop

#     def step(self, action):
#         sarsd = [0,0,0,0,0]
#         ist = copy.deepcopy(self.state)
#         e = self.num_edges
#         g_id = int(np.dot(ist[:e], 2**np.arange(e)))
#         i = self.pointer
#         n_id = copy.deepcopy(i)
#         sarsd[0] = (g_id,n_id)
#         act = action
#         sarsd[1] = act
#         next_state = ist
#         reward = 0
#         if act[0] == i:
#           if self.chaos == 1:
#             cut = list(range(self.num_nodes))
#             cut.remove(act[0])
#             j = random.sample(cut)
#             act[0] = j
#         j = act[0]
#         index = np.zeros(self.obs_dim)
#         index[i+self.num_edges] -= 1
#         index[self.num_edges+j] += 1
#         next_state += index
#         position, kron = self.triangulate(i,j)
#         if position == 0 and self.stop_type == 'pass':
#           self.waiting += 1
#         if position != 0 and act[1] == 1:
#           if hasattr(self, 'waiting'):
#             self.waiting = 0
#           next_state[position-1] = (next_state[position-1] + 1)%2
#           self.adj[i,j] = (self.adj[i,j]+1)%2
#           self.adj[j,i] = (self.adj[j,i]+1)%2
#           self.graph = nx.Graph(self.adj)
#           self.score = self.score_ev(self.graph)
#         self.state = next_state
#         self.score_traj.append(copy.deepcopy(self.score))
#         self.pointer = j
#         self.time += 1
#         done = self.stop_check()
#         if self.rew_type == 'sparse':
#           if done:
#             reward = self.score
#         if self.rew_type == 'sparse_max':
#           if done:
#             reward = max(self.score_traj)
#         if self.rew_type == 'cont':
#           reward = self.score
#         if self.rew_type == 'cont_var':
#           reward = self.score_traj[-1] - self.score_traj[-2]
#         sarsd[2] = reward
#         ng_id = int(np.dot(next_state[:e], 2**np.arange(e)))
#         nn_id = copy.deepcopy(self.pointer)
#         sarsd[3] = (ng_id,nn_id)
#         sarsd[4] = int(done)
#         self.history.append(sarsd)
#         return next_state, reward, done

#     def reset(self):
#         self.state = copy.deepcopy(self.initial_state)
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.score = copy.deepcopy(self.initial_score)
#         self.pointer = 0
#         self.traj = [copy.deepcopy(self.initial_score)]
#         self.time = 0
#         self.done = False
#         self.history = []
#         if hasattr(self, 'waiting'):
#           self.waiting = 0
#         return self.initial_state, self.initial_score, self.done

# class GlobEnv(gym.Env):
#     def __init__(self, num_nodes, score='wagner-1', rew_type='sparse', stop_type='fixed'):
#         super(GlobEnv, self).__init__()
#         self.name = 'global'
#         self.rew_types = {'sparse','sparse_max','cont','cont_var'}
#         self.rew_type = rew_type
#         self.score_fun = score
#         self.num_nodes = num_nodes
#         self.miner = feats_miner(self.num_nodes)
#         self.score_ev = self.miner.invariants[self.score_fun]
#         self.triangles = [(i*(i+1)//2) for i in range(self.num_nodes)]
#         self.num_edges = int((self.num_nodes * (self.num_nodes-1))//2)
#         self.observation_space = gym.spaces.MultiBinary(self.num_edges)
#         self.initial_state = np.zeros(self.num_edges)
#         self.state = copy.deepcopy(self.initial_state)
#         self.action_space = gym.spaces.Discrete(self.num_edges+1)
#         self.reward_range = (-float('inf'), float('inf'))
#         self.initial_adj = np.zeros((self.num_nodes,self.num_nodes))
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.initial_graph = nx.Graph(self.initial_adj)
#         self.graph = copy.deepcopy(self.initial_graph)
#         self.initial_scores = {'sparse': 0, 'sparse_max': 0, 'cont': self.score_ev(self.graph), 'cont_var': self.score_ev(self.graph)}
#         self.initial_score = copy.deepcopy(self.initial_scores[rew_type])
#         self.score = copy.deepcopy(self.initial_score)
#         self.score_traj = [copy.deepcopy(self.initial_score)]
#         self.done = False
#         self.time = 0
#         self.stop_types = {'fixed': 1, 'dynamic': copy.deepcopy(self.target_score), 'pass': 1}
#         self.stop_type = stop_type
#         self.stop_param = copy.deepcopy(self.stop_types[self.stop_type])
#         if self.stop_type == 'pass':
#           self.waiting = 0
#         self.history = []

#     def act_to_indx(self, act):
#       i = 0
#       j = 0
#       while act > self.triangles[j]:
#         j += 1
#       if j != 0:
#         i = act - (self.triangles[j-1] + 1)
#       return i, j

#     def stop_check(self):
#       stop = False
#       if self.stop_type == 'fixed':
#         if self.time == self.stop_param * self.num_edges:
#           stop = True
#       elif self.stop_type == 'dynamic':
#         if self.score_traj[-1] > self.stop_param:
#           stop = True
#       elif self.stop_type == 'pass':
#         if self.waiting == self.stop_param:
#           stop = True
#       return stop

#     def step(self, action):
#         sarsd = [0,0,0,0,0]
#         next_state = copy.deepcopy(self.state)
#         e = self.num_edges
#         sarsd[0] = int(np.dot(next_state, 2**np.arange(e)))
#         sarsd[1] = action
#         i, j = self.act_to_indx(action)
#         if (i,j) == (0,0) and hasattr(self, 'waiting'):
#           self.waiting += 1
#         if (i,j) != (0,0):
#           if hasattr(self, 'waiting'):
#             self.waiting = 0
#           next_state[action-1] = (next_state[action-1] + 1)%2
#           self.adj[i,j] = (self.adj[i,j] + 1)%2
#           self.adj[j,i] = (self.adj[j,i] + 1)%2
#           self.graph = nx.Graph(self.adj)
#         self.state = next_state
#         self.score = self.score_ev(self.graph)
#         self.score_traj.append(copy.deepcopy(self.score))
#         self.time += 1
#         done = self.stop_check()
#         reward = 0
#         if self.rew_type == 'sparse':
#           if done:
#             reward = self.score
#         if self.rew_type == 'sparse_max':
#           if done:
#             reward = max(self.score_traj)
#         if self.rew_type == 'cont':
#           reward = self.score
#         if self.rew_type == 'cont_var':
#           reward = self.score_traj[-1] - self.score_traj[-2]
#         sarsd[2] = reward
#         sarsd[3] = int(np.dot(next_state, 2**np.arange(e)))
#         sarsd[4] = int(done)
#         self.history.append(sarsd)
#         return next_state, reward, done

#     def reset(self):
#         self.state = copy.deepcopy(self.initial_state)
#         self.adj = copy.deepcopy(self.initial_adj)
#         self.score = copy.deepcopy(self.initial_score)
#         self.traj = [copy.deepcopy(self.initial_score)]
#         self.done = False
#         self.time = 0
#         if hasattr(self, 'waiting'):
#           self.waiting = 0
#         self.history = []
#         return self.initial_state, self.initial_score, self.done

