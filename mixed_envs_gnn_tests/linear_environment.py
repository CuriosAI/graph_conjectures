import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LinearEnvironment(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, dense_reward=False, start_with_complete_graph=True, verbose=True):
        super(LinearEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * number_of_nodes
        self.value_fun = value_fun
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2*self.number_of_edges)
        self.dense_reward = dense_reward
        self.start_with_complete_graph = start_with_complete_graph
        self.best_score_ever = -np.Inf
        self.verbose = verbose
        self.reset()
    
    def state_to_observation(self):
        timestep_flattened = np.reshape(self.timestep, self.number_of_edges)
        graph_flattened = np.reshape(self.graph, self.number_of_edges)
        concatenated = np.concatenate((graph_flattened, timestep_flattened))
        return np.copy(concatenated)
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)
        
        self.done = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        self.timestep = np.zeros(shape, dtype=np.int8)
        self.timestep[0,0] = 1
        self.graph = np.ones(shape, dtype=np.int8) if self.start_with_complete_graph else np.zeros(shape, dtype=np.int8)
        self.old_value = self.value_fun(self.graph, self.timestep)
        
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        old_observation = self.state_to_observation()
        self.graph[self.timestep==1] = action
        current_edge_row, current_edge_col = np.nonzero(self.timestep)
        current_edge_row = current_edge_row[0]
        current_edge_col = current_edge_col[0]
        
        if current_edge_row < current_edge_col:
            tmp = current_edge_col
            current_edge_col = current_edge_row
            current_edge_row = tmp
        
        current_edge_col += 1
            
        if(current_edge_col > current_edge_row):
            current_edge_col = 0
            current_edge_row += 1
        if(current_edge_row>= self.number_of_nodes):
            self.done = True
        
        self.timestep[self.timestep==1] = 0
        if not(self.done):
            self.timestep[current_edge_row, current_edge_col] = 1
            self.timestep[current_edge_col, current_edge_row] = 1

        observation = self.state_to_observation()
        new_value = self.value_fun(self.graph, self.timestep)
        if new_value == -np.Inf:
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
        
        info = {}
        return observation, self.last_reward, self.done, False, info
    
    def render(self):
        return # no-op