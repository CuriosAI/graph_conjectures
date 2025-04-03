import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LinearEnvironment(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, init_graph=None, normalize_reward=False, dense_reward=False, check_at_every_step=False, start_with_complete_graph=True, verbose=True, self_loops = False):

        super(LinearEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.self_loops = self_loops
        if self.self_loops:
            self.number_of_edges = number_of_nodes * (number_of_nodes + 1)//2
        else: 
            self.number_of_edges = number_of_nodes * (number_of_nodes - 1)//2
        self.stop = self.number_of_edges
        self.value_fun = value_fun
        self.normalize = normalize_reward
        self.check_every = check_at_every_step
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2*self.number_of_edges)
        self.dense_reward = dense_reward
        self.init = init_graph
        self.start_with_complete_graph = start_with_complete_graph
        self.best_score_ever = -np.Inf
        self.verbose = verbose
        self.graph = None
        self.name = "Linear"
        self.current = [0,0]
        
        self.reset()
    
    def state_to_observation(self):
        graph_flattened = np.zeros(self.number_of_edges)
        timestep_flattened = np.zeros(self.number_of_edges)
        index = 0
        for i in range(self.number_of_nodes):
            if self.self_loops:          
                for j in range(i,self.number_of_nodes):
                    graph_flattened[index] = self.graph[i,j]
                    timestep_flattened[index] = self.timestep[i,j]
                    index +=1
            else:
                for j in range(i+1,self.number_of_nodes):
                    graph_flattened[index] = self.graph[i,j]
                    timestep_flattened[index] = self.timestep[i,j]
                    index += 1
        concatenated = np.concatenate((graph_flattened, timestep_flattened))
        return np.copy(concatenated)
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)
        self.done = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        self.timestep = np.zeros(shape, dtype=np.int8)
        if self.self_loops:
            self.timestep[0,0] = 1
            self.current = [0,0]
        else:
            self.timestep[0,1] = 1
            self.current = [0,1]

        if self.init is not None:
            self.graph = self.init
        else:
            if self.start_with_complete_graph and self.self_loops:
                self.graph = np.ones(shape, dtype=np.int8)
            if self.start_with_complete_graph and not self.self_loops:
                self.graph = np.ones(shape, dtype=np.int8) - np.eye(shape[0], dtype=np.int8)
            else:
                self.graph = np.zeros(shape, dtype=np.int8)
        
        self.old_value = self.value_fun(self.graph, self.normalize)
        
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def get_valid_actions(self):
        return np.array([True, True], dtype=bool)
    
    def step(self, action):
       
        old_observation = self.state_to_observation()
        #self.graph[self.timestep==1] = action
        for i in range(self.number_of_nodes):
            for j in range(i if self.self_loops else i + 1, self.number_of_nodes):
                if self.timestep[i, j] == 1:
                    self.graph[i, j] = action
                    self.graph[j, i] = action

        current_edge_row, current_edge_col = np.nonzero(self.timestep)
        current_edge_row = current_edge_row[0]
        current_edge_col = current_edge_col[0]
        
        if current_edge_row < current_edge_col:
            tmp = current_edge_col
            current_edge_col = current_edge_row
            current_edge_row = tmp
        
        current_edge_col += 1
        if self.self_loops:
            if(current_edge_col > current_edge_row):
                current_edge_col = 0
            current_edge_row += 1
            if(current_edge_row>= self.number_of_nodes):
                self.done = True
        else:
            if(current_edge_col >= current_edge_row):
                current_edge_col = 0
                current_edge_row += 1
            if(current_edge_row>= self.number_of_nodes):
                self.done = True
       
        self.timestep[self.timestep==1] = 0
        if not(self.done):
            self.timestep[current_edge_row, current_edge_col] = 1
            self.timestep[current_edge_col, current_edge_row] = 1
            self.current = [current_edge_row, current_edge_col]   
        observation = self.state_to_observation()
        
        self.last_reward = 0
        new_value = 0
        
        if self.check_every and current_edge_row < self.number_of_nodes:
            new_value = self.value_fun(self.graph, self.normalize)
            if new_value > 1e-12:
                self.done = True
        
        if self.done:
            new_value = self.value_fun(self.graph, self.normalize)
            
        if self.dense_reward:
            self.last_reward = new_value - self.old_value
        if self.done:
            self.last_reward = new_value
        self.old_value = new_value
        
        if new_value > self.best_score_ever:
            self.best_score_ever = new_value
            self.best_score_in_episode = new_value
            func_name = self.value_fun.__name__
            filename = f"./results/best_graphs/best_graph_{self.name}_{self.number_of_nodes}_{func_name}.npy"
            np.save(filename, self.graph)
        if new_value > self.best_score_in_episode:
            self.best_score_in_episode = new_value

        if self.verbose and self.done:
            print(f"best_score_ever={self.best_score_ever}, best_score_in_episode={self.best_score_in_episode}, final_score={new_value}")
        
        info = {}
        return observation, self.last_reward, self.done, False, info
    
    def render(self):
        return
    
