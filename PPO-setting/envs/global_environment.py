import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GlobalEnvironment(gym.Env):
    def __init__(self, number_of_nodes, value_fun, init_graph=None, normalize_reward=False, time_horizon=None, dense_reward=False, check_at_every_step=False, start_with_complete_graph=True, verbose=True, self_loops=False, action_type=None):

        super(GlobalEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"

        # GAME PARAMETERS
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * (number_of_nodes + 1) // 2
        self.action_space = spaces.Discrete(self.number_of_nodes**2)
        self.action_type = action_type
        self.observation_space = spaces.MultiBinary(self.number_of_edges)
        self.init = init_graph
        self.start_with_complete_graph = start_with_complete_graph
        self.self_loops = self_loops
        self.graph = self.init
        self.name = "Global"
        self.value_fun = value_fun
        self.last_reward = 0
        self.normalize = normalize_reward
        self.check_every = check_at_every_step
        self.dense_reward = dense_reward
        self.current = [0,0] 
        if time_horizon is None:
            if self.self_loops:
                self.stop = self.number_of_nodes*(self.number_of_nodes + 1) // 2
            else:
                self.stop = self.number_of_nodes*(self.number_of_nodes - 1) // 2
        else:
            self.stop = time_horizon
        self.verbose = verbose
        self.best_score_ever = -np.Inf

        self.reset()

    def state_to_observation(self):
        graph_flattened = self.graph[np.triu_indices(self.graph.shape[0],k=0)]
        return graph_flattened.flatten()

    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)
        self.done = False
        self.truncated = False
        shape = (self.number_of_nodes, self.number_of_nodes)
        if self.init:
            if self.action_type == None:
              print("ERROR! Action type cannot be inferred\n")
            else:
              self.graph = self.init
        else:
            if self.start_with_complete_graph and self.self_loops:
                self.graph = np.ones(shape, dtype=np.int8)
                self.current = [0,0]
                self.action_type = 0
            elif self.start_with_complete_graph and not self.self_loops:
                self.graph = np.ones(shape, dtype=np.int8) - np.eye(shape[0], dtype=np.int8)
                self.current = [0,1]
                self.action_type = 0
            else:
                self.graph = np.zeros(shape, dtype=np.int8)
                self.current = [0,0]
                self.action_type = 1

        self.timestep_it = 0
        self.old_value = self.value_fun(self.graph, self.normalize)
        self.best_score_in_episode = -np.Inf
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def get_valid_actions(self):
        mask = np.ones(self.number_of_nodes ** 2, dtype=bool) 
        for action in range(self.number_of_nodes ** 2):
            i = action // self.number_of_nodes
            j = action % self.number_of_nodes
            if not self.self_loops:
                if i == j:  
                    mask[action] = False  
        return mask
    
    def step(self, action):
        old_observation = self.state_to_observation()
        info = {}
        i = action // self.number_of_nodes
        j = action % self.number_of_nodes
        if not self.self_loops and i == j:
            print("Invalid move: trying to add self loop\n")
        else:
            self.graph[i,j] = self.action_type
            self.graph[j,i] = self.action_type
            self.timestep_it += 1
            self.current = [i,j]

            if(self.timestep_it >= self.stop):
                self.truncated = True
                self.done = True

            observation = self.state_to_observation()
            self.last_reward = 0
            new_value = 0

            if self.check_every and self.timestep_it < self.stop:
                new_value = self.value_fun(self.graph, self.normalize)
                if new_value > 1e-12:
                    self.done = True
            
            if self.timestep_it >= self.stop:
                new_value = self.value_fun(self.graph, self.normalize)
                self.done = True
                if new_value > 1e-12:
                    self.done = True

            if self.dense_reward:
                self.last_reward = new_value - self.old_value
            if self.done or self.truncated:
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

            if self.verbose and (self.done or self.truncated):
                print(f"best_score_ever={self.best_score_ever}, best_score_in_episode={self.best_score_in_episode}, final_score={new_value}")
            return observation, self.last_reward, self.done, self.truncated, info

    def render(self):
        return