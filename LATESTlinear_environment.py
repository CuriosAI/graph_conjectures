import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

import numpy as np
# print(np.__file__)
# print(gym.__file__)

#class LinearEnvironment(gym.Env):    
class LinEnv(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, normalize_reward = True, dense_reward=False, start_with_complete_graph=True, verbose=True, self_loops=False):
        super(LinEnv, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * (number_of_nodes - 1)//2
        self.row_count = number_of_nodes
        self.column_count = number_of_nodes

        self.value_fun = value_fun
        self.normalize = normalize_reward
        self.dense_reward = dense_reward

        self.action_space = spaces.Discrete(2)
        self.action_size = 2
        self.observation_space = spaces.MultiBinary(2*self.number_of_edges)

        self.start_with_complete_graph = start_with_complete_graph
        self.best_score_ever = -np.Inf
        self.verbose = verbose
        self.self_loops = self_loops
        self.reset()
    
    # CHECK
    def state_to_observation(self):
        timestep_flattened = np.reshape(self.timestep, self.number_of_nodes**2)
        graph_flattened = np.reshape(self.graph, self.number_of_nodes**2)
        concatenated = np.concatenate((graph_flattened, timestep_flattened))
        return np.copy(concatenated)
    
    # CHECK
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)
        
        self.done = False
        self.truncated = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        self.timestep = np.zeros(shape, dtype=np.int8)
        if self.self_loops:
            self.timestep[0,0] = 1
        else:
            self.timestep[0,1] = 1
        
        self.graph = np.ones(shape, dtype=np.int8) if self.start_with_complete_graph else np.zeros(shape, dtype=np.int8)
        # i self loop potrebbero essere considerati, in altre conetture...
        # aggiungiamo un flag
        if self.start_with_complete_graph and self.self_loops:
            self.graph = np.ones(shape, dtype=np.int8)
        elif self.start_with_complete_graph:
            self.graph = np.ones(shape, dtype=np.int8) - np.eye(shape[0], dtype=np.int8)
        else:
            self.graph = np.zeros(shape, dtype=np.int8)
        
        self.old_value = self.value_fun(self.graph, self.normalize)
        
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    # CHECK
    def step(self, action, simulate=False):
        # Affinchè la logica di action vada bene sia nel caso di partenza dal grafo completo
        # che in quello di partenza dal grafo vuoto
        """
        ACTION = 0 --> Non cambiare lo stato dell'arco
        ACTION = 1 --> Cambia lo stato dell'arco (se c'è, togli; se non c'è, metti)
        (Flip)
        """
        if not simulate:
            # Modifica del grafo
            old_observation = self.state_to_observation()
            #if action:
                #self.graph[self.timestep==1] ^= 1
                
            # Aggiornamento del turno
            # current --> indici del nuovo arco
            current_edge_row, current_edge_col = np.nonzero(self.timestep)
            # len(current_edge_row) = len(current_edge_col) = 2
            current_edge_row = current_edge_row[0]
            current_edge_col = current_edge_col[0]
            
            self.graph[current_edge_row,current_edge_col] = 1 - self.graph[current_edge_row,current_edge_col]
            self.graph[current_edge_col,current_edge_row] = self.graph[current_edge_row,current_edge_col]
            # self.graph[self.timestep==1] ^= action
                # action = 0 ---> xor(0,action) = 0
                #                 xor(1,action) = 1
                # action = 1 ---> xor(0,action) = 1
                #                 xor(1,action) = 0
            
            # Siamo SOTTO LA DIAGONALE?
            # Allora, scegli l'altra coppia
            if current_edge_col < current_edge_row:
                current_edge_col = current_edge_col[1]
                current_edge_row = current_edge_row[1]
            # Aggiorna come devi
            if current_edge_col > current_edge_row:
                  current_edge_col += 1
                  if current_edge_col > self.number_of_nodes - 1:
                    current_edge_row += 1
                    current_edge_col = current_edge_row + 1
                    # Abbiamo finito i turni: la partita è troncata
                    if current_edge_col > self.number_of_nodes - 1:
                      self.truncated = True

            # if current_edge_row < current_edge_col:
            #     tmp = current_edge_col
            #     current_edge_col = current_edge_row
            #     current_edge_row = tmp
            
            # current_edge_col += 1
                
            # if(current_edge_col > current_edge_row):
            #     current_edge_col = 0
            #     current_edge_row += 1
            
            # Abbiamo finito i turni: la partita è troncata
                
            
            # Aggiornamento del timestep, sia nella parte sopra la diagonale
            # che in quella sotto
            # Azzera l'arco precedente
            self.timestep[self.timestep==1] = 0
            
            if not(self.done) and not(self.truncated):
                self.timestep[current_edge_row, current_edge_col] = 1
                self.timestep[current_edge_col, current_edge_row] = 1

        observation = self.state_to_observation()
        # -----------------------------------------------------------
        
        # INDICAZIONE DA AGGIUNGERE: la value_fun ha bisogno di due input
        # Abbiamo vinto?
        new_value,info = self.value_fun(self.graph, self.normalize)
        if new_value > 1e-12:
            self.done = True

        self.last_reward = new_value
        return observation, self.last_reward, self.done, self.truncated, info
        
        """if new_value == -np.Inf:
            self.done = True
            observation = old_observation
            new_value = self.old_value
        Penalizzazione di -Inf nel caso di grafi sconnessi. Troppo drammatica
        Inoltre, Andrea non aggiornava il value, in un certo senso... mentre noi si
        """
            
        # X ora non mi interesso di come vengono aggiornati questi attributi
        """self.last_reward = 0
        # reward incrementale, noi non ne faremo uso
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
        #info = {}
    
    # ---
    def render(self):
        nx.draw(nx.Graph(self.graph))
        return # no-op
    
    # ALPHAZERO SUITABILITY
    # ------------------------------------------------------------------
    def get_valid_moves(self):
        valid_moves = np.zeros(self.number_of_edges, dtype=np.int8)
        turn = self.timestep
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

    # ---
    def extract_action(self,encoded_action):
      if encoded_action >= self.number_of_edges:
         return 1
      else:
         return 0
    
    def get_encoded_state(self):
      """   adj --> self.graph
            turn --> self.timestep """
      triu = np.zeros((self.number_of_nodes,self.number_of_nodes),dtype=np.int8)
      for i in range(self.number_of_nodes):
          for j in range(i+1,self.number_of_nodes):
              triu[i,j] = self.graph[i,j]
      encoded_state = np.stack((triu == 0,triu == 1,self.timestep == 0,self.timestep == 1)).astype(np.float32)
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

