import numpy as np
import random
import math
print(np.__version__)
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import copy
torch.manual_seed(0)
from tqdm.notebook import trange

""" This code defines a class called Node for representing nodes in a Monte Carlo Tree Search (MCTS) tree. 
    The Node class has the following attributes:
    # 1. game: A reference to the game environment
    # 2. args: A dictionary containing MCTS algorithm parameters
    # 3. state: The current game state
    # 4. parent: The parent node in the tree
    # 5. action_taken: The action that led to this node
    # 6. prior: A prior probability for this node
    # 7. visit_count: The number of times this node has been visited
    # 8. value_sum: The sum of all values collected from this node """

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    # The is_fully_expanded() method determines whether a node in a Monte Carlo Tree Search (MCTS) tree has been fully expanded.
    # A node is considered fully expanded if it has a child node for every possible action in the current game state.
    # This means that the MCTS has explored all possible moves from the current position, allowing it to make an informed decision about the best action to take.

    def is_fully_expanded(self):
        return len(self.children) > 0

    # The select() method is responsible for selecting the most promising child node to explore next.
    # This decision is crucial as it guides the MCTS algorithm towards potentially valuable regions of the game tree, increasing the likelihood of finding good moves.
    # The method identifies the most promising child node as the one with the highest UCB value.

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    # The get_ucb() method is responsible for calculating the Upper Confidence Bound (UCB) value for a child node.
    # The UCB value is a metric that balances exploration and exploitation,
    # guiding the MCTS algorithm towards potentially valuable regions of the game tree while still giving credit to nodes that have already shown promise.
    # IMPORTANT: The q_value is determined by the point of view of the parent node. Since in this framework a parent node and its child are determined
    # by the action of two different players, a good action for the parent node is a bad action for its opponent, hence we have that
    # q_value = 1 - ....
    # For different applications consider a different definition of the q_value

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
            # In this way the q_value lies between 0 and 1
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    # The expand() method is responsible for creating new child nodes in the game tree.
    # This expansion process ensures that the MCTS has a comprehensive representation of the available moves and their potential outcomes,
    # allowing it to make more informed decisions about the best course of action.

    # 1. Iterate over actions: The method iterates through the list of possible actions for the current game state.
    #    Each action represents a potential move that can be taken from the current position.

    # 2. Create child nodes: For each valid action, the method creates a new child node in the game tree.
    #    The child node inherits the parent node's game state and action information, along with its own visit count and value sum.

    # 3. Assign prior probability: The method assigns a prior probability to each child node.
    #    This probability represents the initial belief about the potential value of the child node, based on the game's rules and the parent node's information.

    # 4. Return child nodes: The method returns the list of newly created child nodes, which represent the expanded portion of the game tree.


    def expand(self, policy):
        for index, prob in enumerate(policy):
            ''' l'action fa il ruolo del player, ma va ricavata
                In questo momento è un indice in {0 a n(n-1) - 1}
                in LinEnv, la posizione su cui agire si ricava dallo state'''
            if index < self.game.number_of_edges:
                action = 1
            else:
                action = 0
            if prob > 0:
                # 1: copiare il gioco del padre
                child_game = copy.deepcopy(self.game)
                # 2 : aggiornare lo stato (fare lo step)
                child_game.step(action) # tutta l'informazione su quanto avvenuto è dentro child_game
                # 3 : crea il nodo figlio
                child = Node(child_game, # gioco
                             self.args,  # args
                             child_game.state, # stato del figlio
                             self,   # nodo padre
                             action, # action_taken
                             prob, # prior = policy
                             # visit_count = 0 <-- DEFAULT
                             )
                self.children.append(child)
        # return child # <-- C'è bisogno di un return? Bisogna controllare
        # Al momento mi sembra di no

    # The backpropagate() method is responsible for updating the value and visit counts of nodes based on the outcome of a simulated rollout.
    # This process allows the MCTS algorithm to incorporate the results of simulated play into its understanding of the game tree,
    # improving its decision-making capabilities over time.

    # 1. Update value: The method starts by updating the value of the current node.
    # This is done by summing the value obtained from the rollout with the previous estimate.

    # 2. Increment visit count: The method then increments the visit count of the current node, indicating that it has been explored one more time.

    # 3. Propagate updates: The method recursively calls itself on the parent nodes of the current node, passing down the updated value and visit count.
    # This propagates the learning signal from the leaf nodes of the game tree back to the root node.

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        #value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)