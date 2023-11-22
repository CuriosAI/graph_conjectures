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
import numpy as np

class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.n_in = observation_space.shape[0]
        self.n_edges = self.n_in // 2
        self.n_nodes = int((1 + np.sqrt(1 + 8*self.n_edges)))//2
        self.n_out = features_dim
        self.gcnconv = GCNConv(in_channels=1, out_channels=self.n_out)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_edges = self.n_in // 2
        graphs = observations[:,:n_edges]
        timesteps = observations[:,n_edges:]
        x = th.ones([self.n_nodes,1], dtype=th.float)
        edge_index = []
        idx = 0
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if(graphs[0,idx]):
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                idx = idx + 1
        edge_index_tensor = th.tensor(edge_index, dtype=th.long)
        return th.concatenate((th.nn.functional.log_softmax(th.nn.functional.relu(self.gcnconv(x, edge_index_tensor))), timesteps))

if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    number_of_nodes=18
    env = WagnerLinearEnvironment(number_of_nodes,dense_reward=True, penalize_star=True)
    total_timesteps = 20_000_000
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(    total_timesteps = 20_000_000)