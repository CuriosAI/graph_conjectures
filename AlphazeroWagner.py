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

from MCTSWagner import MCTS

class AlphaZero:

    """This is the AlphaZero class constructor. It takes four arguments:
    1. model: A neural network model that takes a game state as input and outputs a policy distribution over the possible moves and a value estimate for the current state.
    2. optimizer: An optimizer that updates the weights of the neural network model based on the training data.
    3. game: The game environment that the AlphaZero agent is playing.
    4. args: A dictionary containing the hyperparameters for the AlphaZero algorithm."""

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    # The selfPlay method enables the agent to learn from its own experience.
    # It involves playing games against itself, using the Monte Carlo tree search (MCTS) to select moves and learn from the outcomes of those moves.

    # 1. Initialize the memory buffer: Create an empty list to store the game states, policy probabilities, and outcomes collected during self-play.
    # 2. Set the initial player: Assign the current player to 0 (arbitrary choice) and start the game from the initial state.
    # 3. Perform MCTS search and select a move: For each turn in the game, use the MCTS algorithm to generate a policy distribution over the possible moves.
    #    Select a move based on this policy distribution, applying a temperature hyperparameter to balance exploration and exploitation.
    # 4. Simulate the game: Play out the remaining moves of the game using the selected move and the current policy distribution.
    #    This simulates the game from the perspective of the current player.
    # 5. Record the outcome: Once the game ends, record the outcome (win or loss) associated with the current player.
    #    This outcome will be used to evaluate the effectiveness of the selected move and the policy distribution.
    # 6. Append the game data to the memory buffer: Add the current game state, the selected move, and the recorded outcome to the memory buffer.
    #    This data will be used later for training the neural network model.
    # 7. Repeat the loop: Continue playing games until the desired number of self-play iterations is reached.
    #    This process generates a large dataset of game states and outcomes that can be used to train the neural network model.

    def selfPlay(self):
        out = open('output.txt','a')
        # Initialize a memory buffer to store game states, policy probabilities, and outcomes
        memory = []
        # Set the current player as 0 (arbitrary choice)
        player = 0
        # Initialize the game state and play until a terminal state is reached
        # state = self.game.get_initial_state() (old)
        state = self.game.reset()

        while True:
            # Obtain a policy distribution from the current game state using MCTS
            print('Starting mcts search at state {0}\n'.format(state))


            action_probs = self.mcts.search(state) # LinEnv --> len(action_probs)=2*numero archi
            print('action probs: {0}\n'.format(action_probs))


            # Append the current game state, policy distribution, and player to the memory buffer
            # x l'apprendimento serve solo una metà
            memory.append((state, action_probs[:self.game.number_of_edges]))

            # Simulate a game using the current policy distribution and record the outcome
            temperature_action_probs = action_probs ** (1 / self.args['temperature']) # lungo 2*numero archi
            # perturbazione di action_probs
            print('temperature action probs: {0}\n'.format(temperature_action_probs))

            encoded_action = np.random.choice(self.game.action_size, p=temperature_action_probs/np.sum(temperature_action_probs))

            # fai lo step con l'azione decisa tramite la policy empirica del mcts
            print('Pre action state: {0}\n'.format(self.game.state))
            if isinstance(encoded_action,int):
              self.game.step(self.game.extract_action(encoded_action)) # aggiorna lo stato del game
            else:
              self.game.step(*self.game.extract_action(encoded_action)) # Local & Global
            print('Post action state: {0}\n'.format(self.game.state))

            # Obtain the game outcome from the current state and player
            value, is_terminal = self.game.get_value_and_terminated()
            # If the game has ended, terminate the loop and return the collected memory
            if is_terminal:

                print('terminal, with state {0}\n'.format(self.game.state))# Convert the memory data into a format suitable for training the neural network
                returnMemory = []
                for hist_state, hist_action_probs in memory:
                    # Calculate the outcome for each history entry
                    hist_outcome = value # if hist_player == player else self.game.get_opponent_value(value)
                    # Append the encoded history state, policy probabilities, and outcome to the return memory
                    returnMemory.append((
                        self.game.get_encoded_state(),
                        hist_action_probs,
                        hist_outcome
                    ))
                m = open('memory.txt','a')
                m.write(str(returnMemory))
                m.write('\n')
                m.close()
                return returnMemory

    # The train method is responsible for training the neural network model based on the data collected during self-play. It consists of two main steps:

    # Data Preparation: Shuffle the memory buffer and extract batches of data of the specified size (batch size).
    # Model Training: For each batch of data, forward pass the neural network to generate predictions for the policy probabilities and value estimate.
    #    Calculate the loss using cross-entropy and mean squared error, respectively. Backpropagate the loss through the network to update the weights.

    # More in detail:

    # 1. Shuffle the Memory Buffer: Randomize the order of the game data in the memory buffer to prevent the model from overfitting to specific sequences of states.
    # 2. Extract Batches of Data: Iterate through the shuffled memory buffer and extract batches of data of the specified size (batch size).
    #    Each batch should contain a mix of states, policy probabilities, and outcomes from different self-play games.
    # 3. Forward Pass and Loss Calculation: For each batch of data, pass the game states through the neural network. The network should output two values:
    #    a. Policy Probabilities: A vector of probabilities representing the likelihood of choosing each possible move from the current state.
    #    b. Value Estimate: A scalar value representing the estimated value of the current state, indicating whether the current player is likely to win or lose.
    # 4. Calculate the cross-entropy loss between the predicted policy probabilities and the target policy probabilities from the batch data.
    #    Calculate the mean squared error loss between the predicted value estimate and the target value estimate from the batch data.
    # 5. Backpropagation and Weight Updates: Use the calculated losses to backpropagate through the neural network.
    #    Update the weights of the network in the direction that minimizes the losses.
    #    This process helps the network learn to better predict the policy probabilities and value estimates of game states.
    # 6. Repeat the Training Loop: After processing each batch of data, repeat the process until the specified number of epochs is reached.
    #    This ensures that the neural network is trained with a sufficient amount of data to generalize well to unseen game states.

    def train(self, memory):
        # Shuffle the memory buffer to randomize the order of data samples. This helps prevent overfitting to specific sequences of states.
        random.shuffle(memory)
        # Iterate over batches of data
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            # Extract a batch of data
            sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            #Technical comment: sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]  Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error

            # Convert the batch data into a format suitable for the neural network
            state, policy_targets, value_targets = zip(*sample)
            # Convert the data into tensors and move them to the device (gpu if available or cpu)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # Forward pass to calculate the network outputs
            # out al momento è lungo #archi, e deve restare così:
            # la rete deve aggiornare tutti i suoi pesetti
            # ma: policy_targets viene dai valori salvati in memoria, ed e lungo 2
            # vogliamo ri-allargarlo a un vettore lungo #archi
            # in cui la prob delle azioni non valide è 0
            # così da confrontare distribuzioni sullo stesso numero di azioni possibili
            out_policy, out_value = self.model(state)

            # Calculate the loss for the policy and value heads
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            # Backward pass to calculate the gradients of the loss with respect to the model parameters.
            loss = policy_loss + value_loss
            # Zero the gradients of the model parameters.
            self.optimizer.zero_grad()
            loss.backward()
            # Update the model parameters using the gradients calculated from backpropagation.
            self.optimizer.step()

    # The learn method is the main training loop of the AlphaZero algorithm.
    # It iterates over a specified number of iterations, each of which consists of two phases: self-play and training.

    # Self-play phase:

    # 1. Initialize an empty memory buffer: This buffer will be used to store the data collected during self-play games.
    # 2. Set the model to evaluation mode: This ensures that the model is not updated during self-play,
    #    allowing it to focus on exploring different play strategies without being biased by previous training.
    # 3. Play multiple self-play games: For the specified number of self-play iterations, play games against itself using the MCTS algorithm to generate moves.
    # 4. Append the game data to the memory buffer: After each game, append the collected game states, policy probabilities, and outcomes to the memory buffer.

    # Training phase:

    # 1. Set the model to training mode: This allows the model to update its weights based on the data collected during self-play.
    # 2. Train the model for multiple epochs: For the specified number of epochs, iterate over the memory buffer and perform forward pass,
    #    loss calculation, backward propagation, and weight updates. This process refines the model's ability to predict policy probabilities and value estimates.
    # 3. Save the model and optimizer state: After each iteration, save the model's state and optimizer's state to disk.
    #    This allows for resuming training later or using the trained model for evaluation or gameplay.

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            # Initialize an empty memory buffer for storing self-play data.
            memory = []
            # Set the model to evaluation mode for exploration
            self.model.eval()
            # Play multiple self-play games
            i = 1
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                # Perform self-play and collect game data
                print('\n\n SELF PLAY N.{0}\n'.format(i))
                memory += self.selfPlay()
                i += 1
            # Set the model to training mode for refining policy and value estimates
            self.model.train()
            # Train the model for multiple epochs using the collected data
            for epoch in trange(self.args['num_epochs']):
                # Train the neural network model using the memory buffer
                self.train(memory)
            # Save the model and optimizer state for later use
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game.number_of_nodes}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game.number_of_nodes}.pt")