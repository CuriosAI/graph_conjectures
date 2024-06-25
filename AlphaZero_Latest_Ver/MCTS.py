import numpy as np
import random
import math
print(np.__version__)
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm import trange
from Node import Node
from linear_environment import LinEnv

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
        #print('Root parent state {0}\n'.format(state))
        #print('Root game state {0}\n'.format(root.game.state_to_observation()))
        #print('Root parent {0}\n'.format(root.parent))

        # Get an initial policy distribution from the neural network model
        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(), device=self.model.device).unsqueeze(0)
        )
        #print(f"Policy before perturbation: {policy}")
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        #print(f"Policy before perturbation - softmax: {policy}")
        
        # Add a small amount of exploration noise to the policy distribution
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet( [self.args['dirichlet_alpha']] * 2*self.game.number_of_edges ) 
        #print('Policy shape = {0}\n'.format(policy.shape))
        #print('Perturbed policy: {0}\n'.format(policy))

        # Mask probabilities related to invalid moves as 0
        valid_moves = self.game.get_valid_moves().flatten()
        # flatten x righe
        policy *= valid_moves
        #print('Masked Total policy: {0}\n'.format(policy))

        # Complement
        #policy = np.concatenate((add_policy,np.where(add_policy > 0, 1-add_policy, 0)))
        #print('Total policy: {0}\n'.format(policy))

        # Normalize the adapted policy
        policy /= np.sum(policy)
        #print('Normalized Total Perturbed policy: {0}\n'.format(policy))

        # Expand the root node to create child nodes for each valid action
        root.expand(policy)

        # Perform multiple simulations for each child node
        j = 0
        for search in range(self.args['num_searches']): # Il numero di simulazioni è un iperparametro
            print('Search n.{0}\n'.format(search))
            node = root
            #print('Root children: {0}\n'.format(len(root.children)))

            # Select a child node to explore based on the Upper Confidence Bound (UCB) formula
            # questa è la parte che permette ad mcts di scorrere i figli
            # E DI ANDARE IN PROFONDITA'!!
            # Più le search aumentano, più percorro un ramo
            # Infatti, deep passa da 1 a 3! (Senza passare per 2)
            while node.is_fully_expanded(): # while the given node has children
                node = node.select() # select the best child based on the UCB value
                j += 1
            # Simulate a rollout from the current node to a terminal state

            #print('Deep {0}\n'.format(j))
            #print('Input state {0}\n'.format(state))
            #print('MCTS self.game state {0}\n'.format(self.game.state_to_observation()))
            #print('Node root state {0}\n'.format(root.game.state_to_observation()))
            #print('Node parent state {0}\n'.format(node.parent.state))
            #print('Node state {0}\n'.format(node.state))

            '''Occhio: cosa c'è in game? Lo stato terminale? Se si, ok'''
            # value, is_terminal = self.game.get_value_and_terminated()
            # x com'è ora, il game dell'MCTS non si sposta...

            # for that given child return the reward and the win/loss info

            # value, is_terminal = node.game.get_value_and_terminated()
            _, value, win, truncated, _ = node.game.step(0,simulate=True)
            is_terminal = win or truncated
            # If the simulation has not reached a terminal state, obtain a policy distribution from the neural network model and expand the current node
            if not is_terminal:

                policy, value = self.model(torch.tensor(node.game.get_encoded_state(), device=self.model.device).unsqueeze(0)
                )
                # Get probs
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                #print('Value NN: {0}\n'.format(value))
                # Add
                policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                 * np.random.dirichlet( [self.args['dirichlet_alpha']] * 2*self.game.number_of_edges )
                #print('Node Perturbed policy: {0}\n'.format(policy))

                # Mask probabilities related to invalid moves as 0
                valid_moves = node.game.get_valid_moves().flatten() # (?)
                policy *= valid_moves
                #print('Node Masked policy: {0}\n'.format(policy))

                # Normalize the adapted policy
                policy /= np.sum(policy)
                #print('Node Normalized Total Perturbed policy: {0}\n'.format(policy))

                node.expand(policy)

            # Backpropagate the value obtained from the simulation, the value obtained from the NN is used only for intermediate states to reach a terminal node
            node.backpropagate(value)
            #print('Node visit count: {0}\n'.format(node.visit_count))

        """Adesso, in uno qualsiasi dei 3 giochi, action_taken
        sta nell'insieme {0, ... , 2m-1}
        A seconda del gioco, alcuni valori in questo insieme non verranno mai assunti
        quindi le action_probs (di questi figli mai raggiunti) varranno sempre 0"""
        action_probs = np.zeros(2*self.game.number_of_edges)
        print('len children = {0}'.format(len(root.children)))
        for child in root.children:
            #print('root.game.state: {0} \n'.format(root.state))
            #print('timestep: {0} \n'.format(root.game.timestep))
            #print('Child.game.state: {0} \nChild state: {1}\nAction taken: {2}'.format(child.game.state_to_observation(),child.state,child.action_taken))
            
            action_probs[child.action_taken]=child.visit_count

        if np.sum(action_probs):
          action_probs /= np.sum(action_probs)
        #print('action probs all children: {0} \n'.format(action_probs))
        return action_probs