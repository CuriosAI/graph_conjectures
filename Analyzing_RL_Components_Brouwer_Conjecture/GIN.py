##########################################################################################
##                                                                                      ##
##      ANALYZING RL COMPONENTS FOR WAGNER'S FRAMEWORK VIA BROUWER'S CONJECTURE         ##
##                                                                                      ##
##########################################################################################

""" 
This script contains the architectures used in our experiments with Wagner's algorithm.
You can find:
- the GIN and GCN structures confronted on the laplacian dataset.
- a module called policy_layers, used to predict policies after feature extraction
  (done with GIN or other customizable nets).
- a module called action_predictor to combined a given feature extractor to policy_layers for
  predicting policies in case of Linear, Local and Global environments. Must be adapted
  to other environments in case of usage.
"""

import torch as th
import torch.nn as nn

from torch_geometric.nn.models import MLP
import numpy as np

import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv import GINEConv

from torch_geometric.nn.models import JumpingKnowledge

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GIN(nn.Module):
    """
        n_nodes: number of nodes of the input graphs
        device: the device on which the model will be instantiated (cpu/cuda)
        n_conv: number of convolutions to perform on the input. The default value has been used in
                in mentioned experiments
        red: a parameter that disables/enables the net portion for eigenvalues prediction in case
             red=True/red=False.
    """
    def __init__(self, n_nodes, device, n_conv=4, red=False):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_edges = n_nodes*n_nodes
        self.device = device
        self.reduction = red

        # GINEConv layers
        mlp = MLP(in_channels=1, hidden_channels=32, out_channels=32, num_layers=4).to(device)
        self.gines = [GINEConv(mlp, edge_dim=1).to(device)]
        for i in range(n_conv-1):
            mlp = MLP(in_channels=32, hidden_channels=32, out_channels=32, num_layers=4).to(device)
            self.gines.append(GINEConv(mlp, edge_dim=1).to(device))
        
        # Aggregation layer
        self.jk = JumpingKnowledge("max").to(device)
        
        # Node features
        self.x = th.ones((self.n_nodes, 1), device=device)

        # Edges' indexes to be passed in GINEConv layers
        edge_index = th.zeros((2, self.n_nodes * self.n_nodes), device=device, dtype=th.long)
        for i in range(self.n_edges):
            edge_index[0, i] = i // self.n_nodes
            edge_index[1, i] = i % self.n_nodes
        self.edge_index = edge_index
    
        # MLP for eigenvalues' prediction
        self.eigenvalues = MLP([self.n_nodes*32,self.n_nodes*128,self.n_nodes*64,self.n_nodes*32,self.n_nodes]).to(device)

    def forward(self, observations) -> th.Tensor:
        
        if isinstance(observations, np.ndarray):
            observations = th.Tensor(observations)

        # Different tensor's shapes must be treaten differnlty
        # before passing it to the convolutional layers
        if isinstance(observations, th.Tensor):
            if observations.dim() == 1:
                # one single observation
                n_stuff = 1

            elif observations.dim() == 2:
                n_stuff = observations.size(0)
                
            else:
                n_stuff = observations.size(0)*observations.size(1)
            
            edge_information = th.reshape(observations, (n_stuff, self.n_nodes**2, 1)).to(self.device).type(th.float32)

        # Creating the dataloader to merge all the graph in a batch        
        data_list = []

        for it in range(n_stuff):
            data = Data(x=self.x, edge_index=self.edge_index, edge_attr=edge_information[it,:,:])
            data_list.append(data)

        dataloader = DataLoader(data_list, batch_size=n_stuff)
        batched_data = next(iter(dataloader))

        # Feature extraction phase
        out = batched_data.x
        outs = []
        for gine in self.gines:
            out = gine(out, batched_data.edge_index, batched_data.edge_attr)
            outs.append(out)
        out = self.jk(outs)
        out = th.reshape(out,(n_stuff,self.n_nodes*32))
        
        # Termination or eigenvalues' prediction
        if self.reduction:
            return out
        else:
            eigenvalues = self.eigenvalues(out)
            return eigenvalues

class GCN(nn.Module):
    """
        n_nodes: number of nodes of the input graphs
        device: the device on which the model will be instantiated (cpu/cuda)
        red: a parameter that disables/enables the net portion for eigenvalues prediction in case
             red=True/red=False.
    """
    def __init__(self, n_nodes, device, red=False):
        super(GCN, self).__init__()
 
        self.n_nodes = n_nodes
        self.n_edges = n_nodes*n_nodes
        self.device = device
        self.reduction = red

        # Node features
        self.x = th.ones((self.n_nodes, 1), device=device)

        # Edges' indexes to be passed in GINEConv layers
        edge_index = th.zeros((2, self.n_nodes * self.n_nodes), device=device, dtype=th.long)
        for i in range(self.n_edges):
            edge_index[0, i] = i // self.n_nodes
            edge_index[1, i] = i % self.n_nodes
        self.edge_index = edge_index

        # GCNConv layers
        # in_channels equals to the dimension of nodes' features
        in_channels = 1
        hidden_channels = 33
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Aggregation layers
        self.jk = JumpingKnowledge("max").to(device)
        
        # Fully Connected layers for eigenvalues' prediction
        out_channels = n_nodes
        self.fc1 = nn.Linear(hidden_channels*n_nodes, 64*n_nodes)
        self.fc2 = nn.Linear(64*n_nodes, out_channels) 

    def forward(self, observations) -> th.Tensor:
        
        if isinstance(observations, np.ndarray):
            observations = th.Tensor(observations)

        # Different tensor's shapes must be treaten differnlty
        # before passing it to the convolutional layers
        if isinstance(observations, th.Tensor):
            if observations.dim() == 1:
                # one single observation
                n_stuff = 1

            elif observations.dim() == 2:
                n_stuff = observations.size(0)
                
            else:
                n_stuff = observations.size(0)*observations.size(1)
        
        # Creating the dataloader to merge all the graph in a batch
        data_list = []
        obs_tensors = observations
        edge_information = th.reshape(obs_tensors, (n_stuff, self.n_edges, 1)).to(self.device).type(th.float32)

        for it in range(n_stuff):
            data = Data(x=self.x, edge_index=self.edge_index, edge_attr=edge_information[it,:,:])
            data_list.append(data)

        dataloader = DataLoader(data_list, batch_size=n_stuff)
        batched_data = next(iter(dataloader))
        
        # Feature extraction phase
        out = batched_data.x
        outs = []

        out = self.conv1(out, batched_data.edge_index, batched_data.edge_attr)
        outs.append(out)
        out = self.conv2(out, batched_data.edge_index, batched_data.edge_attr)
        outs.append(out)
        out = self.conv3(out, batched_data.edge_index, batched_data.edge_attr)
        outs.append(out)

        # Aggregation phase
        outs = self.jk(outs)

        # Termination or eigenvalues' prediction
        if self.reduction:
            return outs
        else:
            outs = F.relu(outs)
            outs = F.dropout(outs, p=0.4, training=self.training)
            outs = th.reshape(outs,(n_stuff,self.n_nodes*32))
            eigenvals = self.fc2(self.fc1(outs))
            return eigenvals


# The net used for policy prediction after the GIN feature extractor
class policy_layers(nn.Module):
    """
        n_nodes: number of nodes of the input graphs
        n_actions: number of game's available actions
        device: the device on which the model will be instantiated (cpu/cuda)
    """
    def __init__(self, game, n_nodes, n_actions, device):

        super().__init__()
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.device = device

        if game.name == "Linear":
            self.input_dim = 32*n_nodes + game.number_of_edges
        
        if game.name == "Local":
            self.input_dim = 33*n_nodes
        
        if game.name == "Global":
            self.input_dim = 32*n_nodes
        
        else:
            raise Exception("Unavailable game")
        
        self.layers = MLP([self.input_dim,self.n_nodes*128,self.n_nodes*64,self.n_nodes*32,self.n_actions]).to(device)
        self.proba = nn.Softmax()

    def forward(self, graph: th.Tensor) -> th.Tensor:
        
        if isinstance(graph, np.ndarray):
            graph = th.Tensor(graph)
        
        if len(graph.size()) == 1:
            graph = th.reshape(graph,(1, len(graph)))
            
        Y = self.layers(graph)
        return self.proba(Y)

""" Action predictor used to interact with the environments
    This class can be used with both the GIN feature extractor or a customizable one.
    If no feature extractor is give, the action_predictor istantiates a fully connected network.
    Default values are thought to work in combination with the GIN network."""
class action_predictor(nn.Module):
    
    def __init__(self, feature_extr, game, n_nodes, n_actions,
                 device, neurons=[128,64], obslen=None):
        super().__init__()
        
        try:
            self.name = game.name
        except Exception as e:
            print(f"Error {e} \n game does not have name attribute.")

        self.device = device
        self.n_nodes = n_nodes
        
        # Feature extractor
        self.feature_extr = feature_extr
        if feature_extr is None:
            # Define an MLP
            self.action_pred = nn.Sequential(
                                nn.Linear(obslen, neurons[0]),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
            for i in range(1, len(neurons)):
                self.action_pred = self.action_pred.append(nn.Linear(neurons[i - 1], neurons[i]))
                self.action_pred = self.action_pred.append(nn.ReLU())
                self.action_pred = self.action_pred.append(nn.Dropout(0.2))
            
            self.action_pred = self.action_pred.append(nn.Linear(neurons[-1], n_actions))
       
        else:
            # Uses the feature extractor given, without re-training
            # its parameters
            for param in self.feature_extr.parameters():
                param.requires_grad = False
            self.action_pred = policy_layers(game, n_nodes, n_actions, device)

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = th.Tensor(observation).to(self.device)
        if isinstance(observation, th.Tensor):
            n_stuff = observation.size(0)

        if len(observation.size()) == 1:
            if self.feature_extr is None:
                return self.action_pred(observation)
            else:
                # Re-shaping in several cases
                if self.name == "Linear":
                    adj_flatten = observation[:self.n_nodes**2]
                    features = self.feature_extr(adj_flatten)
                    if len(features.size()) > 1:
                        features = th.flatten(features)
                    return self.action_pred(th.cat((features, observation[self.n_nodes**2:])))
            
                if self.name == "Local":
                    adj_flatten = observation[:self.n_nodes**2]
                    features = self.feature_extr(adj_flatten)
                    if len(features.size()) > 1:
                        features = th.flatten(features)
                    return self.action_pred( th.cat( (features, observation[self.n_nodes**2:]),
                                                dim = 1) )
            
                if self.name == "Global":
                    return self.action_pred(self.feature_extr(observation))
        
        if observation.dim() == 3:
            i,j,k = observation.size()
            observation = th.reshape(observation, (i*j,k))
            n_stuff = i*j
        
        if self.feature_extr is None:
            return self.action_pred(observation)
        else:   
            if self.name == "Linear":
                adj_flatten = observation[0:n_stuff, 0:self.n_nodes**2]
                features = self.feature_extr(adj_flatten)
                return self.action_pred( th.cat( (features, observation[0:n_stuff, self.n_nodes**2:]),
                                                dim = 1) )
            
            if self.name == "Local":
                adj_flatten = observation[0:n_stuff, 0:self.n_nodes**2]
                features = self.feature_extr(adj_flatten)
                return self.action_pred( th.cat( (features, observation[0:n_stuff, self.n_nodes**2:]),
                                                dim = 1) )
            
            if self.name == "Global":
                return self.action_pred(self.feature_extr(observation))
