import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LocalEnvironment(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, dense_reward=False, start_with_complete_graph=True, verbose=True):
        super(LocalEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * number_of_nodes
        self.value_fun = value_fun
        self.action_space = spaces.Discrete(2*self.number_of_nodes)
        self.observation_space = spaces.MultiBinary(self.number_of_edges + self.number_of_nodes)
        self.dense_reward = dense_reward
        self.start_with_complete_graph = start_with_complete_graph
        self.best_score_ever = -np.Inf
        self.verbose = verbose
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
        shape = (self.number_of_nodes, self.number_of_nodes)
        self.graph = np.ones(shape, dtype=np.int8) if self.start_with_complete_graph else np.zeros(shape, dtype=np.int8)
        self.position = 0
        self.old_value = self.value_fun(self.graph,self.position)
        self.timestep_it = 0
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        old_observation = self.state_to_observation()

        flip = action // self.number_of_nodes
        new_position = action % self.number_of_nodes

        if flip > 0:
            self.graph[self.position,new_position] = 1 - self.graph[self.position,new_position]
            self.graph[new_position,self.position] = self.graph[self.position,new_position]
        self.position = new_position
        self.timestep_it += 1
        
        if(self.timestep_it >= self.number_of_edges):
            self.done = True

        observation = self.state_to_observation()
        new_value = self.value_fun(self.graph, self.position)
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