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



class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        
    def __repr__(self):
        return "TicTacToe"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    # a state is in the form of a 3x3 matrix or board M where Mij = -1, 1 if players -1 and 1 
    # have put a piece in the ij cell respectively and 0 if no piece has been placed yet.
    # an action is determined by an integer between 0 and 8 since the board M has 9 entries in total.
    # the player could be 1 or -1.
    
    # given a state, an action and the player, the get_next_state method places a player in the entry defined by the action.
    # This is the board configuration wrt to the actions:
    # !0,1,2|
    # |3,4,5|
    # |6,7,8|

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    # given a state, this method returns an array of valid moves from that given state
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    # The overall logic of this method is to check if the sum of the chosen cell's elements along any row, column, 
    #or diagonal matches the expected sum for a winning line, considering the player's mark. 
    #If any of these conditions are met, the code returns True, indicating a win. Otherwise, it returns False.

    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    
    # The get_value_and_terminated method takes the current game state and the last played action as inputs and returns a tuple containing two values:
    # 1. Value: This indicates the reward or outcome of the game for the player who just made the move (action). We could get 1 if we win the game and 0 otherwise
    #2. Terminated: This is a boolean flag indicating whether the game has ended after the last move. We could get False if the game has not ended yer and True in case
    # of a win or a tie.

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
    # The get_encoded_state method converts the current game state (represented as a NumPy array) into a format suitable for input to a neural network.
    # Note that for other applications the number of stacked planes should be modified.
