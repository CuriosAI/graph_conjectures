import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#torch.manual_seed(0)
from tqdm import trange

#from AlphazeroParallelWagner import AlphaZeroParallel
#from WagnerEnv import WagnerLin
from AlphazeroWagner import AlphaZero
from envs import LinEnv
# from ResnetWagner import ResNet
from GNN import GNN

#game = LinEnv(18,reward='wagner') ''' <-- Env '''
game = LinEnv(18,reward='wagner') 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ResNet(game, 6, 64, device) ''' <-- GNN '''
model = GNN(game.number_of_nodes, 5, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 15,
    'num_iterations': 5,
    'num_selfPlay_iterations': 10,
    #'num_parallel_games': 500,
    'num_epochs': 5,
    'batch_size': 25,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

'''args = {
    'C': 2,
    'num_searches': 10000,
    'num_iterations': 5,
    'num_selfPlay_iterations': 1000,
    'num_parallel_games': 500,
    'num_epochs': 5,
    'batch_size': 512,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}'''

# alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()