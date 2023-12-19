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

from MCTS import MCTS

class ResNet(nn.Module):

# This class defines a ResNet neural network architecture for AlphaZero.
    
# It consists of a start block, a series of residual blocks, a policy head, and a value head.
# It takes four arguments:
# 1. game (Game): The game environment for which the network is being trained.
# 2. num_resBlocks (int): The number of residual blocks in the network.
# 3. num_hidden (int): The number of hidden units in each residual block.
# 4. device (torch.device): The device to use for computations (CPU or GPU).

    def __init__(self, game, num_resBlocks, num_hidden, device):
        # Initialize the base class and set the device
        super().__init__()
        self.device = device
        # Start block: Converts the game state into a feature representation
        self.startBlock = nn.Sequential(
            # Convolutional layer to extract features from the state
            # The startBlock maps the state into a 3-plane feature space. 
            # --------> Change this value for a different representation
            # The padding value for the convolutional layers in the startBlock and valueHead of the ResNet class is set to 1. 
            # This means that the feature maps in the resulting representations will have the same size as the input representations. 
            # This padding strategy helps to preserve the spatial information in the game state representations, 
            # which is crucial for capturing the relationships between different tiles on the board.
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            # Batch normalization to stabilize the network
            nn.BatchNorm2d(num_hidden),
            # ReLU activation to introduce non-linearity
            nn.ReLU()
        )
        # Residual blocks: Aggregate information from multiple game states
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        # Policy head: Predicts the probability distribution over possible moves
        self.policyHead = nn.Sequential(
            # Convolutional layer to extract additional features from the feature representation
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            # Batch normalization to stabilize the network
            nn.BatchNorm2d(32),
            # ReLU activation to introduce non-linearity
            nn.ReLU(),
            # Flatten the feature representation
            nn.Flatten(),
            # Fully connected layer to predict the policy probabilities
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        # Value head: Predicts the estimated value of the current state
        self.valueHead = nn.Sequential(
            # Convolutional layer to extract additional features from the feature representation
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
             # Batch normalization to stabilize the network
            nn.BatchNorm2d(3),
            # ReLU activation to introduce non-linearity
            nn.ReLU(),
            # Flatten the feature representation
            nn.Flatten(),
            # Fully connected layer to predict the value estimate
            nn.Linear(3 * game.row_count * game.column_count, 1),
            # Tanh activation to ensure the value estimate lies within [-1, 1]
            nn.Tanh()
        )
        # Move the network to the specified device
        self.to(device)

    def forward(self, x):
        # Calculates the policy and value predictions for a given game state. It takes one argument:
        # 1. x (torch.Tensor): The game state representation as a 3D tensor.
        # The output is a (torch.Tensor, torch.Tensor): The predicted policy probabilities and value estimate for the current state.

        # Apply the start block to convert to a feature representation
        x = self.startBlock(x)
        # Iterate through the residual blocks
        for resBlock in self.backBone:
            # Apply each residual block
            x = resBlock(x)
        # Extract policy and value predictions from the final layer
        policy = self.policyHead(x)
        value = self.valueHead(x)
        # Return the policy and value predictions
        return policy, value

# The ResBlock class represents a residual block, which is a common building block in convolutional neural networks (CNNs). 
# Residual blocks are used to aggregate information from multiple layers in the network and introduce non-linearity. 
# This can help to improve the network's ability to learn complex patterns and relationships from the data.

# In this specific implementation, the ResBlock class takes an input tensor x and applies:
# two convolutional layers, two batch normalization layers, and an additional ReLU activation function. 
# The residual block then adds the output of the second convolutional layer to the input tensor x and passes the sum through another ReLU activation function. 
# This process helps the network to learn more complex patterns from the data by allowing it to bypass the direct path through the residual block.

# It takes one argument:
# 1. num_hidden (int): The number of hidden units in each convolutional layer of the residual block.

# It has the following attributes:
# 1.  conv1 (nn.Conv2d): The first convolutional layer in the residual block.
# 2. bn1 (nn.BatchNorm2d): The first batch normalization layer in the residual block.
# 3. conv2 (nn.Conv2d): The second convolutional layer in the residual block.
# 4. bn2 (nn.BatchNorm2d): The second batch normalization layer in the residual block.

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        # Initialize convolutional layers and batch normalization layers.

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        # Define the residual block
        residual = x
        # Apply the first convolutional layer and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        # Apply the second convolutional layer and batch normalization
        x = self.bn2(self.conv2(x))
        # Add the residual and apply a ReLU activation
        x += residual
        x = F.relu(x)
        # Return the updated feature representation
        return x