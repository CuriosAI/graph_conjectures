# from stable_baselines3 import PPO, A2C, DQN
# Import x fare RL, non ci dovrebbero servire
# from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
# RMSProp --> training, quì non ci dovrebbe servire
from stable_baselines3.common.env_util import make_vec_env
# ? Questo cos'è?

# from wagner_linear_environment import WagnerLinearEnvironment
from envs import LinEnv

import torch as th
import torch.nn as nn
from gymnasium import spaces

# from stable_baselines3 import PPO
#from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.models import MLP
import numpy as np

import torch.nn.functional as F
from torch_geometric.nn.conv import GINEConv

from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.pool import global_max_pool, global_mean_pool, global_add_pool

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from datetime import datetime

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


#class CustomGNN(BaseFeaturesExtractor):
# cambiare la classe base
class GNN(nn.Module): # ?
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    # input: stato LinEnv (vettore)
    def __init__(self,n_nodes,n_conv,device):
        # input del costruttore: direttamente n_nodes,n_edges
        # super().__init__(observation_space, features_dim=32)
        super().__init__()
        #self.n_in = observation_space.shape[0]
        self.n_nodes = n_nodes
        self.n_edges = n_nodes*n_nodes
        self.device = device
        # input : matrice di adiacenza
        mlp = MLP(in_channels=1, hidden_channels=32, out_channels=32, num_layers=4).to(device)
        self.gines = [GINEConv(mlp, edge_dim=1).to(device)]
        for i in range(n_conv-1):
            mlp = MLP(in_channels=32, hidden_channels=32, out_channels=32, num_layers=4).to(device)
            self.gines.append(GINEConv(mlp, edge_dim=1).to(device))
        self.jk = JumpingKnowledge("max").to(device)    # prendi il massimo di tutti i valori sui nodi
        self.x = th.ones((self.n_nodes, 1), device=device)

        # Archi x le GNN: matrice 2 * n_edges, colonne (tipo): (0 | 1), (0 | 2) ... (i | j) ...
        edge_index = th.zeros((2, self.n_nodes * self.n_nodes), device=device, dtype=th.long)
        for i in range(self.n_edges):
            edge_index[0, i] = i // self.n_nodes
            edge_index[1, i] = i % self.n_nodes
        self.edge_index = edge_index

        # policy e value head
        self.policyHead = MLP(in_channels=self.n_nodes*32, hidden_channels=self.n_nodes*128, out_channels=self.n_nodes*(self.n_nodes-1)//2, num_layers=3, norm="LayerNorm").to(device)
        self.valueHead = MLP(in_channels=self.n_nodes*32, hidden_channels=self.n_nodes*128, out_channels=1, num_layers=3, norm="LayerNorm").to(device)
        self.normalize_value = nn.Tanh().to(device)


    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_stuff = observations.shape[0]
        data_list = []

        edge_information = th.reshape(observations, (n_stuff, self.n_edges, 1)).to(self.device).type(th.float32)

        for it in range(n_stuff):
            data = Data(x=self.x, edge_index=self.edge_index, edge_attr=edge_information[it,:,:])
            data_list.append(data)

        # print(f"construction: {(end - start).total_seconds()}")
        dataloader = DataLoader(data_list, batch_size=n_stuff)
        batched_data = next(iter(dataloader))
        out = batched_data.x
        outs = []
        for gine in self.gines:
            out = gine(out, batched_data.edge_index, batched_data.edge_attr)
            # matrice dim (n_stuff * n_nodes) * 32 (in generale: n_nodi * n_ feature)
            outs.append(out)
        out = self.jk(outs) # (n_stuff * n_nodes) * 32
        out = th.reshape(out,(n_stuff,self.n_nodes*32))

        policy = self.policyHead(out)
        value = self.normalize_value(self.valueHead(out))

        return policy,value
        # print(f"inference: {(end-start).total_seconds()}")







        # serviva per il LinEnv
        # observations = th.reshape(observations, (n_stuff, 2, self.n_edges)).to(device)
        # l'idea era: mettere il timestep del LinEnv sotto lo stato
        # per ogni elemento del minibatch