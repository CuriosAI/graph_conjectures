from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike 
from stable_baselines3.common.env_util import make_vec_env

from wagner_linear_environment import WagnerLinearEnvironment
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
        self.n_nodes = int((1 + np.sqrt(1 + 8*self.n_edges)))//2
        self.gin = GAT(2,256,8,out_channels=256, jk="max", edge_dim=2,v2=True).to(device)   
        mlp = MLP(in_channels=2,hidden_channels=256,out_channels=256,num_layers=4).to(device)
        self.gines = [GINEConv(mlp,edge_dim=2).to(device)]
        for i in range(1):
            mlp= MLP(in_channels=256,hidden_channels=256,out_channels=256,num_layers=4).to(device)
            self.gines.append(GINEConv(mlp,edge_dim=2).to(device))
        self.jk = JumpingKnowledge("max").to(device)
        self.x = th.ones((self.n_nodes,2),device=device)

        edge_index = th.zeros((2,self.n_nodes*self.n_nodes),device=device,dtype=th.long)
        for i in range(self.n_edges):
            edge_index[0,i] = i // self.n_nodes
            edge_index[1,i] = i % self.n_nodes  
        self.edge_index = edge_index          

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_stuff = observations.shape[0]
        observations = th.reshape(observations,(n_stuff,2,self.n_edges)).to(device)
                
        data_list = []
        
        edge_information = th.reshape(observations[:,:,:],(n_stuff,2,self.n_edges))
        edge_features = th.zeros((n_stuff,self.n_nodes*self.n_nodes,2),device=device)
        k=0
        for i in range(self.n_nodes):
                for j in range(i):
                    edge_features[:,i*self.n_nodes + j,:] = edge_information[:,:,k]
                    edge_features[:,j*self.n_nodes + i,:] = edge_information[:,:,k]
                    k = k +1
        
        for it in range(0,n_stuff):
            data = Data(x=self.x, edge_index=self.edge_index,edge_attr=edge_features[it,:,:])
            data_list.append(data)
            
        # print(f"construction: {(end - start).total_seconds()}")
        dataloader = DataLoader(data_list,batch_size=n_stuff)
        batched_data = next(iter(dataloader))
        # out = self.gin(batched_data.x, batched_data.edge_index,batched_data.edge_attr)
        out = batched_data.x
        outs = []
        for gine in self.gines:
            out = gine(out, batched_data.edge_index, batched_data.edge_attr)
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
    
    number_of_nodes=25
    env = WagnerLinearEnvironment(number_of_nodes,dense_reward=True, penalize_star=True)
    total_timesteps = 200_000
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=1e-4,n_steps=2048,batch_size=256)
    model.learn(total_timesteps = 200_000)