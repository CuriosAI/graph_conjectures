import numpy as np
import random
import math
print(np.__version__)
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm.notebook import trange

from AlphazeroParallel import AlphaZeroParallel
from MCTSParallel import MCTSParallel
from Tictactoe import TicTacToe
from ConnectFour import ConnectFour
from Resnet import ResNet
from MCTS import MCTS

game = TicTacToe()
player = 1

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 4, 64, device)
model.load_state_dict(torch.load("model_7.pt", map_location=device))
model.eval()

mcts = MCTS(game, args, model)

state = game.get_initial_state()


while True:
    print(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)