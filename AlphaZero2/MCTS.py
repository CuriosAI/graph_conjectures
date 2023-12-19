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

from Node import Node

class MCTS:
    
    # This is the MCTS class constructor. It takes three arguments:

    #1. game: The game instance
    #2. args: A dictionary containing the MCTS algorithm parameters
    #3. model: The neural network model used to evaluate game states
        
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    # The search() method:

    # 1. Creates a root node for the MCTS search.
    # 2. Obtains an initial policy distribution from the neural network model and add exploration noise.
    # 3. Applies the policy distribution to obtain a list of valid moves.
    # 4. Expands the root node to create child nodes for each valid action.
    # 5. Performs multiple simulations for each child node using the select(), get_value_and_terminated(), and backpropagate() methods.
    # 6. Returns the probability distribution of actions from the root node.    
        
    @torch.no_grad()
    def search(self, state):
        # Create a root node for the MCTS search
        root = Node(self.game, self.args, state, visit_count=1)
        # Get an initial policy distribution from the neural network model
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        # Add a small amount of exploration noise to the policy distribution
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        # Apply the policy distribution to obtain a list of valid moves
        valid_moves = self.game.get_valid_moves(state)
        # Mask probabilities related to invalid moves as 0
        policy *= valid_moves
        # Normalize the adapted policy
        policy /= np.sum(policy)
        # Expand the root node to create child nodes for each valid action
        root.expand(policy)
        
        # Perform multiple simulations for each child node
        for search in range(self.args['num_searches']):
            node = root
            # Select a child node to explore based on the Upper Confidence Bound (UCB) formula
            while node.is_fully_expanded():
                
                node = node.select()
            # Simulate a rollout from the current node to a terminal state  
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            # If the simulation has not reached a terminal state, obtain a policy distribution from the neural network model and expand the current node
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
            # Backpropagate the value obtained from the simulation    
            node.backpropagate(value)    
            
        # Return the probability distribution of actions from the root node  
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        