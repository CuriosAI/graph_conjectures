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

from NodeWagner import Node


class MCTS:
    """ This is the MCTS class constructor. It takes three arguments:

     1. game: The game instance
     2. args: A dictionary containing the MCTS algorithm parameters
     3. model: The neural network model used to evaluate game states"""

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    # The search() method:

    # 1. Creates a root node for the MCTS search.
    # 2. Obtains an initial policy distribution from the neural network model and adds exploration noise.
    # 3. Applies the policy distribution to obtain a list of valid moves.
    # 4. Expands the root node to create child nodes for each valid action.
    # 5. Performs multiple simulations  for each child node using the select(), get_value_and_terminated(), and backpropagate() methods.
    # 6. Returns the probability distribution of actions from the root node.

    @torch.no_grad()
    def search(self, state):
        # Create a root node for the MCTS search
        root = Node(self.game, self.args, state, visit_count=1)
        # Control prints
        # print('Root parent state {0}\n'.format(state))
        # print('Root game state {0}\n'.format(root.game.state))
        # print('Root parent {0}\n'.format(root.parent))

        # Get an initial policy distribution from the neural network model
        add_policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(), device=self.model.device).unsqueeze(0)
        )
        add_policy = torch.softmax(add_policy, axis=1).squeeze(0).cpu().numpy()
        # Add a small amount of exploration noise to the policy distribution
        add_policy = (1 - self.args['dirichlet_epsilon']) * add_policy + self.args['dirichlet_epsilon'] \
                     * np.random.dirichlet([self.args['dirichlet_alpha']] * (self.game.action_size // 2))
        # print('Perturbed policy: {0}\n'.format(add_policy))

        # Mask probabilities related to invalid moves as 0
        valid_moves = self.game.get_valid_moves()  # (?)
        add_policy *= valid_moves
        # print('Masked add_policy: {0}\n'.format(add_policy))

        # Complement
        policy = np.concatenate((add_policy, np.where(add_policy > 0, 1 - add_policy, 0)))
        # print('Total policy: {0}\n'.format(policy))

        # Normalize the adapted policy
        policy /= np.sum(policy)
        # print('Normalized Total Perturbed policy: {0}\n'.format(policy))

        # Expand the root node to create child nodes for each valid action
        root.expand(policy)

        # Perform multiple simulations for each child node
        # j = 0
        for search in range(self.args['num_searches']):  # Il numero di simulazioni è un iperparametro
            #print('Search n.{0}\n'.format(search))
            node = root

            # Select a child node to explore based on the Upper Confidence Bound (UCB) formula
            while node.is_fully_expanded():  # while the given node has children
                node = node.select()  # select the best child based on the UCB value
                #j += 1

            # Simulate a rollout from the current node to a terminal state

            # Control prints
            # print('Deep {0}\n'.format(j))
            # print('Input state {0}\n'.format(state))
            # print('MCTS self.game state {0}\n'.format(self.game.state))
            # print('Node root state {0}\n'.format(root.game.state))
            # print('Node parent state {0}\n'.format(node.parent.state))
            # print('Node state {0}\n'.format(node.state))

            # for that given child return the reward and the win/loss info
            value, is_terminal = node.game.get_value_and_terminated()

            # If the simulation has not reached a terminal state, obtain a policy distribution from the neural network model and expand the current node
            if not is_terminal:
                add_policy, value = self.model(
                    torch.tensor(node.game.get_encoded_state(), device=self.model.device).unsqueeze(0)
                    )
                # Get probs
                add_policy = torch.softmax(add_policy, axis=1).squeeze(0).cpu().numpy()

                # Add
                add_policy = (1 - self.args['dirichlet_epsilon']) * add_policy + self.args['dirichlet_epsilon'] \
                             * np.random.dirichlet([self.args['dirichlet_alpha']] * (self.game.action_size // 2))
                # print('Node Perturbed policy: {0}\n'.format(add_policy))

                # Mask probabilities related to invalid moves as 0
                valid_moves = node.game.get_valid_moves()  # (?)

                add_policy *= valid_moves
                # print('Node Masked add_policy: {0}\n'.format(add_policy))

                # Complement
                policy = np.concatenate((add_policy, np.where(add_policy > 0, 1 - add_policy, 0)))
                # print('Node Total policy: {0}\n'.format(policy))

                # Normalize the adapted policy
                policy /= np.sum(policy)
                # print('Node Normalized Total Perturbed policy: {0}\n'.format(policy))

                node.expand(policy)

            # Backpropagate the value obtained from the simulation, the value obtained from the NN is used only for intermediate states to reach a terminal node
            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        # print('len children = {0}'.format(len(root.children)))
        # Calcolo della policy empirica
        # SPECIFICO PER IL LINENV
        # Andrebbe modificato per renderlo generale
        for child in root.children:

            timestep = root.state[self.game.number_of_edges:]
            # State control
            # print('root.game.state: {0} \n'.format(root.state))
            # print('timestep: {0} \n'.format(timestep))
            # print('Child.game.state: {0} \nChild state: {1}\nAction taken: {2}'.format(child.game.state, child.state,
            #                                                                            child.action_taken))
            i = np.argmax(timestep)

            if child.action_taken == 1:
                action_probs[i] = child.visit_count
            else:
                action_probs[i + self.game.number_of_edges] = child.visit_count

        if np.sum(action_probs):
            action_probs /= np.sum(action_probs)
        return action_probs