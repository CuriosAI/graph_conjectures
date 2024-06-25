import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

class LocalEnv(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, stopping_time, normalize_reward=True, dense_reward=False, start_with_complete_graph=True, verbose=True, self_loops=False):
        super(LocalEnv, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * number_of_nodes
        self.row_count = number_of_nodes
        self.column_count = number_of_nodes

        self.value_fun = value_fun
        self.normalize = normalize_reward
        self.dense_reward = dense_reward

        # Partendo dal nodo corrente, ho 2*self.number_of_nodes azioni, nel caso di self_loops
        # 2*self.number_of_nodes - 1 azioni, altrimenti
        # tuttavia, conviene lasciare lo spazio delle azioni sempre 2*self.number_of_nodes azioni
        # faremo attenzione nello step
        self.action_space = spaces.Discrete(2*self.number_of_nodes)
        self.observation_space = spaces.MultiBinary(self.number_of_edges + self.number_of_nodes)
        if self_loops:
            self.action_size = 2*self.number_of_nodes
        else:
            self.action_size = 2*(self.number_of_nodes - 1)
        
        self.start_with_complete_graph = start_with_complete_graph
        self.best_score_ever = -np.Inf
        self.verbose = verbose
        self.self_loops = self_loops
        self.stop = stopping_time
        self.reset()
    
    def state_to_observation(self):
        graph_flattened = np.reshape(self.graph, self.number_of_edges)
        position_one_hot = np.zeros((self.number_of_nodes,),dtype=np.int8)
        position_one_hot[self.position] = 1
        res = np.concatenate((graph_flattened, position_one_hot))
        return np.copy(res)
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)

        self.done = False
        self.truncated = False

        shape = (self.number_of_nodes, self.number_of_nodes)
        if self.start_with_complete_graph and self.self_loops:
            self.graph = np.ones(shape, dtype=np.int8)
        elif self.start_with_complete_graph:
            self.graph = np.ones(shape, dtype=np.int8) - np.eye(shape[0], dtype=np.int8)
        else:
            self.graph = np.zeros(shape, dtype=np.int8)
        #self.graph = np.ones(shape, dtype=np.int8) if self.start_with_complete_graph else np.zeros(shape, dtype=np.int8)
        
        self.position = 0 # <-- position è un nodo
        self.old_value = self.value_fun(self.graph,self.normalize)
        self.timestep_it = 0 # <-- numero di turni
        
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        
        return observation, info
    
    def step(self, action, simulate=False):
        if not simulate:
            old_observation = self.state_to_observation()

            # action in [0:n] --> flip = 0 (lascia invariato)
            # action in [n:2n] --> flip = 1 (lascia invariato)
            """ posizione = i
            action == 0 : lascia l'arco 0,i com'è
             ....
              action == n-1 : .... n,i
            action == n : flippa l'arco 0,i ...  
            action == 2n - 1 : flippa l'arco n,i"""
            flip = action // self.number_of_nodes
            new_position = action % self.number_of_nodes
            if flip > 0:
                self.graph[self.position,new_position] = 1 - self.graph[self.position,new_position]
                self.graph[new_position,self.position] = self.graph[self.position,new_position]
            
            self.position = new_position
            self.timestep_it += 1
            
            if(self.timestep_it >= self.stop):
                self.truncated = True
        # -----------------------------------
        # Se siamo in simulazione, calcola solo lo score
        # Tutta la parte di simulazione è fatta altrove
        observation = self.state_to_observation()
        new_value,info = self.value_fun(self.graph, self.normalize)
        if new_value > 1e-12:
            self.done = True
        
        self.last_reward = new_value
        return observation, self.last_reward, self.done, self.truncated, info
        """ if new_value == -np.Inf:
            self.done = True
            observation = old_observation
            new_value = self.old_value
        self.last_reward = 0
        if self.dense_reward:
            self.last_reward = new_value - self.old_value
        elif self.done:
            self.last_reward = new_value
        self.old_value = new_value
        
        
        if new_value > self.best_score_ever:
            self.best_score_ever = new_value
        if new_value > self.best_score_in_episode:
            self.best_score_in_episode = new_value

        if self.verbose and self.done:
            print(f"best_score_ever={self.best_score_ever}, best_score_in_episode={self.best_score_in_episode}, final_score={new_value}")
        """
        
    def render(self):
        nx.draw(nx.Graph(self.graph))
        return # no-op
    
    # ALPHAZERO SUITABILITY
    # ------------------------------------------------------------------

    # {0-1,0-2,0-3,1-2,1-3,2-3}
    # position = 0 --> allora voglio valid[0] = valid[1] = valid[2] = 1 --> {0 ... number_of_edges/2}
    # position = 1 --> valid[0] = valid[3] = valid[4] = 1 
    # position = 2 --> valid[1] = valid[3] = valid[5] = 1
    def get_valid_moves(self):
        valid_moves = np.zeros(self.number_of_edges, dtype=np.int8)
        turn = np.zeros((self.number_of_nodes,self.number_of_nodes), dtype=np.int8)
        turn[self.position,:] = 1
        turn[:,self.position] = 1
        if not self.self_loops:
            turn[self.position,self.position] = 0
        i = 0
        j = 1
        for k in range(self.number_of_edges):
            if turn[i,j] == 1:
                valid_moves[k] = 1
            j += 1
            if j >= self.number_of_nodes:
                i += 1
                j = i+1
            if i >= self.number_of_nodes - 1:
                break

        return np.concatenate((valid_moves,valid_moves))

    # ? La usiamo ancora ??
    def extract_action(self,encoded_action):
        # n = 4, m = 6
        # encoded_action è in {0, .. , 11 = 2m-1}
        # encoded_action = 10
        # le action x lo step variano in {0, .. , 7 = 2n-1}
        # riceviamo un numero da 0 a 2m-1
        # vogliamo un numero da 0 a 2n-1, che ci dica:
        # -> su quale estremo agire
        # -> cosa fare (flip?)
        action = 0
        # Sono nel caso di flip
        if encoded_action >= self.number_of_edges:
            action += self.number_of_nodes
            encoded_action -= self.number_of_edges
        row = 0
        # Calculate the maximum action index for the current row
        max_action_for_row = self.number_of_nodes - 1  # The last possible action index in the first row
        while encoded_action >= max_action_for_row:
            row += 1
            max_action_for_row += (self.number_of_nodes - row - 1)
        # Calculate the column based on the row and action
        col_offset = encoded_action - (max_action_for_row - (self.number_of_nodes - row - 1))
        column = row + 1 + col_offset
        
        if column == self.position:
            return action+row
        elif row == self.position:
            return action+column
        else:
            return -1 # Hope this will never happen

        
    
    def get_encoded_state(self):
        # Individuiamo tutte le posizioni accessibili nel turno corrente
        turn = np.zeros((self.number_of_nodes,self.number_of_nodes), dtype=np.int8)
        turn[self.position,:] = 1
        turn[:,self.position] = 1
        if not self.self_loops:
            turn[self.position,self.position] = 0
        encoded_state = np.stack((self.graph == 0,self.graph == 1,turn == 0,turn == 1)).astype(np.float32)
        #print(f"Len of encoded_state = {len(encoded_state)}")

      # This is needed in case of parallel execution of Alphazero
        if len(encoded_state.shape) == 2:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
            
"""      # def register_linenv(number_of_nodes):
        def register_linenv(number_of_nodes, normalize_reward):
        # id=f'LinEnv-{number_of_nodes}_nodes-normalize_{normalize_reward}'
        gym.register(
        # id=id,
        id='LinEnv-v0',  # this name is hard-coded in rl_zoo3/hyperparams/ppo.yml, we cannot change it
        entry_point='envs:LinEnv',
        kwargs={'number_of_nodes': number_of_nodes, 'normalize_reward': normalize_reward}
        ) """

