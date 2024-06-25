from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.env_util import make_vec_env

from linear_environment import LinearEnvironment
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.models import GIN, MLP
import numpy as np


import torch.nn.functional as F
from torch_geometric.nn.conv import GINEConv

from torch_geometric.nn.models import GIN, GCN, PNA,GAT, JumpingKnowledge
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.pool import global_max_pool, global_mean_pool, global_add_pool

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from datetime import datetime
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

import networkx as nx
import numpy as np

def wagner_value(graph, timestep):
    n = graph.shape[0]
    nxgraph = nx.Graph(graph)
    for i in range(n):
        if nxgraph.has_edge(i,i):
            nxgraph.remove_edge(i,i)
    # print(nxgraph.edges)
    if not nx.is_connected(nxgraph): #or nx.is_isomorphic(nxgraph, self.nxstar)):
        score = -np.Inf
    else:
        constant = 1 + np.sqrt(n - 1)
        lambda_1 = max(np.real(nx.adjacency_spectrum(nxgraph)))
        mu = len(nx.max_weight_matching(nxgraph,maxcardinality=True))
        score = constant - (lambda_1 + mu)
    
    return score


class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=256)
        self.n_in = observation_space.shape[0]
        self.n_edges = self.n_in // 2
        self.n_nodes = int((np.sqrt(self.n_edges)))
        # self.gin = GAT(2,256,8,out_channels=256, jk="max", edge_dim=2,v2=True).to(device)   
        mlp = MLP(in_channels=2,hidden_channels=256,out_channels=256,num_layers=4).to(device)
        self.gines = [GINEConv(mlp,edge_dim=2).to(device)]
        for i in range(4):
            mlp= MLP(in_channels=256,hidden_channels=256,out_channels=256,num_layers=4).to(device)
            self.gines.append(GINEConv(mlp,edge_dim=2).to(device))
        self.jk = JumpingKnowledge("max").to(device)
        self.x = th.ones((self.n_nodes,2),device=device)
        
        edge_index = th.zeros((2,self.n_edges),device=device,dtype=th.long)
        for i in range(self.n_edges):
            edge_index[0,i] = i // self.n_nodes
            edge_index[1,i] = i % self.n_nodes  
        self.edge_index = edge_index          

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_stuff = observations.shape[0]
        observations = th.reshape(observations,(n_stuff,2,self.n_edges)).to(device)
        # print(f"observations.shape={observations.shape}")
        data_list = []
        
        edge_information = th.reshape(observations[:,:,:],(n_stuff,2,self.n_edges))
        
        for it in range(0,n_stuff):
            data = Data(x=self.x, edge_index=self.edge_index,edge_attr=edge_information[it,:,:])
            data_list.append(data)
            
        # print(f"construction: {(end - start).total_seconds()}")
        dataloader = DataLoader(data_list,batch_size=n_stuff)
        batched_data = next(iter(dataloader))
        # out = self.gin(batched_data.x, batched_data.edge_index,batched_data.edge_attr)
        out = batched_data.x
        outs = []
        for gine in self.gines:
            edge_attr = th.reshape(batched_data.edge_attr, (2,n_stuff*self.n_edges))
            edge_attr = th.transpose(edge_attr,0,1)
            # if n_stuff > 1:
            #     print(f"out.shape={out.shape}, edge_index.shape={batched_data.edge_index.shape}, edge_attr.shape={batched_data.edge_attr.shape} edge_attr={batched_data.edge_attr}")
            out = gine(out, batched_data.edge_index, edge_attr)
            outs.append(out)
        out = self.jk(outs)
        out_mean = global_mean_pool(out, batched_data.batch)                  
        # print(f"inference: {(end-start).total_seconds()}")
        return out_mean

if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(),
    )
    
    number_of_nodes=18
    env = LinearEnvironment(number_of_nodes,wagner_value,dense_reward=True,start_with_complete_graph=True)
    total_timesteps = 200_000
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=1e-4,n_steps=2048,batch_size=256)
    model.learn(total_timesteps = 200_000)