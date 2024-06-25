import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GlobalEnvironment(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, dense_reward=False, start_with_complete_graph=True, verbose=True):
        super(GlobalEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * number_of_nodes
        self.value_fun = value_fun
        self.action_space = spaces.Discrete(self.number_of_edges)
        self.observation_space = spaces.MultiBinary(self.number_of_edges)
        self.dense_reward = dense_reward
        self.start_with_complete_graph = start_with_complete_graph
        self.verbose = verbose
        self.best_score_ever = -np.Inf
        self.reset()
    
    def state_to_observation(self):
        graph_flattened = np.reshape(self.graph, self.number_of_edges)
        return np.copy(graph_flattened)
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)

        self.done = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        self.graph = np.ones(shape, dtype=np.int8) if self.start_with_complete_graph else np.zeros(shape, dtype=np.int8)
        self.timestep_it = 0
        self.old_value = self.value_fun(self.graph)
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        old_observation = self.state_to_observation()
        i = action // self.number_of_nodes
        j = action % self.number_of_nodes
        
        self.graph[i,j] = 1 - self.graph[i,j]
        self.graph[j,i] = self.graph[i,j]
        self.timestep_it += 1
        
        if(self.timestep_it >= self.number_of_edges):
            self.done = True

        observation = self.state_to_observation()
        new_value = self.value_fun(self.graph)
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