import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm.notebook import trange

from AlphazeroParallel import AlphaZeroParallel
from Alphazero import AlphaZero
from Tictactoe import TicTacToe
from ConnectFour import ConnectFour
from Resnet import ResNet

game = TicTacToe()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 1000,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()