import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlipEnvironment(gym.Env):    
    def __init__(self, number_of_nodes, value_fun, init_graph=None, normalize_reward=False, time_horizon=None, dense_reward=False, check_at_every_step=False, start_with_complete_graph=True, verbose=True, self_loops=False):
        super(FlipEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * (number_of_nodes-1) //2
        self.value_fun = value_fun
        self.normalize = normalize_reward
        self.check_every = check_at_every_step
        self.action_space = spaces.Discrete(self.number_of_edges)
        self.observation_space = spaces.MultiBinary(self.number_of_nodes*self.number_of_nodes)
        self.dense_reward = dense_reward
        self.last_reward = 0
        self.init = init_graph
        self.start_with_complete_graph = start_with_complete_graph
        self.self_loops = self_loops
        if time_horizon is None:
            if self.self_loops:
                self.stop = self.number_of_nodes*(self.number_of_nodes + 1) // 2
            else:
                self.stop = self.number_of_nodes*(self.number_of_nodes - 1) // 2
        else: 
            self.stop = time_horizon
        self.verbose = verbose
        self.current = [0,0]
        self.best_score_ever = -np.Inf
        self.reset()
    
    def state_to_observation(self):
        graph_flattened = np.copy(self.graph)
        return graph_flattened.flatten()
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)

        self.done = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        if self.init:
            self.graph = self.init
        else:
            if self.start_with_complete_graph and self.self_loops:
                self.graph = np.ones(shape, dtype=np.int8)
                self.current = [0,0]
            elif self.start_with_complete_graph:
                self.graph = np.ones(shape, dtype=np.int8) - np.eye(shape[0], dtype=np.int8)
                self.current = [0,1]
            else:
                self.graph = np.zeros(shape, dtype=np.int8)
                self.current = [0,0]

        self.timestep_it = 0
        self.timestep = np.zeros(shape, dtype=np.int8)
        self.old_value = self.value_fun(self.graph, self.normalize)
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        old_observation = self.state_to_observation()
        info = {}
        i = action // self.number_of_nodes
        j = action % self.number_of_nodes

        self.current = [i,j]

        if not self.self_loops and i == j:
            print("Invalid move: trying to add self loop\n")
            return old_observation, self.last_reward, self.done, False, info
        else:
            self.graph[i,j] = 1 - self.graph[i,j]
            self.graph[j,i] = 1 - self.graph[j,i]

            self.timestep_it += 1
            
            if(self.timestep_it >= self.stop):
                self.done = True

            observation = self.state_to_observation()
            new_value = self.value_fun(self.graph, self.normalize)

            if self.check_every and self.timestep_it < self.stop:
                if new_value > 1e-12:
                    self.done = True

            self.last_reward = 0
            if self.dense_reward:
                self.last_reward = new_value - self.old_value
            if self.done:
                self.last_reward = new_value
            self.old_value = new_value
            
            if new_value > self.best_score_ever:
                self.best_score_ever = new_value
            if new_value > self.best_score_in_episode:
                self.best_score_in_episode = new_value

            #if self.verbose and self.done:
            #    print(f"best_score_ever={self.best_score_ever}, best_score_in_episode={self.best_score_in_episode}, final_score={new_value}")
            
            return observation, self.last_reward, self.done, False, info
    
    def render(self):
        return
