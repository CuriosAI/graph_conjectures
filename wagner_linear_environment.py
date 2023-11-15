import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import networkx as nx
from gymnasium import spaces

class WagnerLinearEnvironment(gym.Env):
    ACTION_REMOVE = 0 #remove the edge
    ACTION_INSERT = 1 #add the edge
    
    def __init__(self, number_of_nodes, normalize_reward=True, render_mode="none"):
        super(WagnerLinearEnvironment, self).__init__()
        assert isinstance(number_of_nodes, int), "'number_of_nodes' is not an integer"
        
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.number_of_nodes = number_of_nodes
        self.number_of_edges = number_of_nodes * (number_of_nodes -1)//2
        self.observation_space = spaces.MultiBinary(2*self.number_of_edges)
        self.normalize_reward = normalize_reward #Normalize rewards when training, not when evaluating
        self.reset() #here we set the properties timestep and graph and state
    
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
        self.timestep = np.zeros(self.number_of_edges, dtype=np.int8)
        self.timestep[0] = 1
        self.graph = np.zeros(self.number_of_edges, dtype=np.int8)
        self.nxgraph = self.compute_nx_graph()
        self.last_reward = 0
        
        info = {}
        return np.concatenate((self.graph,self.timestep)), info
    
    def step(self, action):
        edge_index = np.argmax(self.timestep)
        self.graph[edge_index] = action
        
        if edge_index < self.number_of_edges - 1:
            self.timestep[edge_index] = 0
            self.timestep[edge_index + 1] = 1
            self.last_reward = 0.0
        else:
            self.timestep[edge_index] = 0 # The terminal state has timestep part all zero
            self.done = True
            self.nxgraph = self.compute_nx_graph()
            if not nx.is_connected(self.nxgraph):
                self.last_reward = -self.number_of_edges*2 if self.normalize_reward else -5.0 # penalty to be used when the conjecture holds only for connected graphs. this normalization assumes that other rewards are > -1
            else:
                constant = 1 + np.sqrt(self.number_of_nodes - 1)
                lambda_1 = max(np.real(nx.adjacency_spectrum(self.nxgraph)))
                mu = len(nx.max_weight_matching(self.nxgraph,maxcardinality=True))
                self.last_reward = constant - (lambda_1 + mu)
        
        info = {}
        if self.done:
            print(self.last_reward)
            if self.render_mode == "human":
                self.render()
        return np.concatenate((self.graph, self.timestep)), self.last_reward, self.done, False, info
        
    def render(self):
        _, ax = plt.subplots()
        pos = nx.spring_layout(self.nxgraph)
        title_string = f"wagner1 score = {self.last_reward}"
        ax.set_title(title_string)
        nx.draw(self.nxgraph, pos=pos, ax=ax, with_labels=True, node_color='lightyellow', font_color='black', edgecolors='black')
        plt.show()
        
        
        
        
        
    
    