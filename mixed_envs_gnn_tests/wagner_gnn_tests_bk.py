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
from torch_geometric.nn.models import GIN, GCN, PNA,GAT
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.pool import global_max_pool, global_mean_pool, global_add_pool



device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=16)
        self.n_in = observation_space.shape[0]
        self.n_edges = self.n_in // 2
        self.n_nodes = int((1 + np.sqrt(1 + 8*self.n_edges)))//2
        n = self.n_edges
        self.gin = GAT(2,4,4,out_channels=16, jk="max")    

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_stuff = observations.shape[0]
        observations = th.reshape(observations,(n_stuff,self.n_edges,2))
        out_mean = th.zeros((n_stuff,16),device=device)
        
        for it in range(0,n_stuff):
            f_in = th.reshape(observations[it,:,:],(self.n_edges,2))
            
            n = self.n_edges
            e = n*n
            edge_index = th.zeros((2,e),dtype=th.int64).to(device)
            
            for i in range(0,e):
                edge_index[0,i] = i // n
                edge_index[1,i] = i % n
            edges_sorted = sort_edge_index(edge_index,sort_by_row=False)

            out = self.gin(f_in, edges_sorted)
            batch = th.zeros((f_in.shape[0],),dtype=th.int64, device=device)
            out_mean[it,:] = global_mean_pool(out, batch)                  
        # print(out_mean.shape)
        return out_mean

if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(),
    )
    
    number_of_nodes=10
    env = WagnerLinearEnvironment(number_of_nodes,dense_reward=True, penalize_star=True)
    total_timesteps = 200_000
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps = 200_000)