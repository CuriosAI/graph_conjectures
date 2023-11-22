import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import networkx as nx
from gymnasium import spaces

class WagnerLinearEnvironment(gym.Env):
    ACTION_REMOVE = 0 #remove the edge
    ACTION_INSERT = 1 #add the edge
    
    def __init__(self, number_of_nodes, dense_reward=False, penalize_star=False):
        super(WagnerLinearEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.number_of_nodes = number_of_nodes
        
        self.number_of_edges = number_of_nodes * (number_of_nodes -1)//2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(2*self.number_of_edges)
        self.episode_number = 0
        self.total_steps = 0
        self.best_wagner_score = -np.Inf
        self.dense_reward = dense_reward
        self.penalize_star = penalize_star
        self.nxstar = nx.star_graph(self.number_of_nodes-1)
        self.reset() # here we set the properties timestep and graph
    
    def compute_nx_graph(self):
        nxgraph = nx.empty_graph(self.number_of_nodes)
        k = 0
        for i in range(self.number_of_nodes):
            for j in range(i+1, self.number_of_nodes):
                if self.graph[k]:
                    nxgraph.add_edge(i,j)
                k = k +1
        return nxgraph
    
    def reset(self, *, seed= None, options = None):
        super().reset(seed=seed, options=options)
        
        self.done = False
        self.episode_number = self.episode_number + 1
        self.timestep = np.zeros(self.number_of_edges, dtype=np.int8)
        self.timestep[0] = 1
        self.graph = np.zeros(self.number_of_edges, dtype=np.int8)
        self.nxgraph = self.compute_nx_graph()

        self.last_reward = 0
        self.last_wagner_score = self.compute_wagner_score(self.nxgraph)
        self.best_wagner_score_in_episode = -np.Inf
        
        info = {}
        return np.concatenate((self.graph,self.timestep)), info
    
    def compute_wagner_score(self, nxgraph):
        if not nx.is_connected(nxgraph) or (self.penalize_star and nx.is_isomorphic(nxgraph, self.nxstar)):
            last_wagner_score = -self.number_of_nodes*2       
        else:
            constant = 1 + np.sqrt(len(self.nxgraph.nodes) - 1)
            lambda_1 = max(np.real(nx.adjacency_spectrum(self.nxgraph)))
            mu = len(nx.max_weight_matching(self.nxgraph,maxcardinality=True))
            last_wagner_score = constant - (lambda_1 + mu)
        return last_wagner_score
    
    def step(self, action):
        edge_index = np.argmax(self.timestep)
        self.graph[edge_index] = action        
        self.nxgraph = self.compute_nx_graph()
        
        if edge_index < self.number_of_edges - 1:
            self.timestep[edge_index] = 0
            self.timestep[edge_index + 1] = 1
        else:
            self.timestep[edge_index] = 0 # The terminal state has timestep part all zero
            self.done = True
        
        old_wagner_score = self.last_wagner_score
        
        self.last_wagner_score = self.compute_wagner_score(self.nxgraph)
        
        if self.last_wagner_score > self.best_wagner_score_in_episode:
            self.best_wagner_score_in_episode = self.last_wagner_score
        if self.last_wagner_score > self.best_wagner_score:
            self.best_wagner_score = self.last_wagner_score
            self.best_nxgraph = self.nxgraph.copy()
            self.best_graph = np.copy(self.graph)
        if self.last_wagner_score > 1e-12:
            self.timestep[edge_index] = 0
            self.done = True
            self.render()

        self.total_steps = self.total_steps + 1
        self.last_reward = 0
        
        if self.done:
            self.last_reward = self.last_wagner_score
            print(f"episode:{self.episode_number}, total_steps:{self.total_steps}, best_score:{self.best_wagner_score}, best_wagner_score_in_episode:{self.best_wagner_score_in_episode}, last_score:{self.last_wagner_score}")
            if self.render_mode == "human":
                self.render()
        
        if self.dense_reward:
            self.last_reward = -old_wagner_score + self.last_wagner_score
        
        info = {}
        return np.concatenate((self.graph, self.timestep)), self.last_reward, self.done, False, info
    
    def render_graph(self, nxgraph, score):
        _, ax = plt.subplots()
        pos = nx.spring_layout(nxgraph)
        title_string = f"wagner1 score = {score}"
        ax.set_title(title_string)
        nx.draw(nxgraph, pos=pos, ax=ax, with_labels=True, node_color='lightyellow', font_color='black', edgecolors='black')
        plt.show()
    
    def render_best_graph(self):
        self.render_graph(self.best_nxgraph, self.best_wagner_score)
        
    def render(self):
        self.render_graph(self.nxgraph, self.last_wagner_score)    