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

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game # inizializzato al game di AlphaZero: quando AlphaZero avanza,
                         # le search partiranno da nuovi stati (?)
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames, iter):
        #print(f"states shape: {states.shape}")
        # tutti gli spGames, all'inizio, contengono la stessa partita
        # sarà sempre così?
        #encoded_states = spGames[0].game.get_encoded_state()

        encoded_states = np.stack([s.game.get_encoded_state() for s in spGames])
        #print(f"encoded_states = {encoded_states}")
        
        #for i in range(1,len(spGames)):
            # shape = (n_nodes,n_nodes,len(spGames))
        #    encoded_states = np.stack((encoded_states,spGames[i].game.get_encoded_state()))
        #print(f"Size of encoded_states : {encoded_states.shape}")

        policy, _ = self.model(torch.tensor(encoded_states, device=self.model.device))
        #print(f"Policy shape: {policy.size}") # size attesa: len(spGames) x 2
        #print(f"Policy 0 before perturbation: {policy[0]}")
        #print(f"Policy 1 before perturbation: {policy[1]}")

        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        #print(f"Policy 0 before perturbation - softmax: {policy[0]}")
        #print(f"Policy 1 before perturbation - softmax: {policy[1]}")
        #print(f"Policy 1 before shape 0: {policy}")
        #print(f"{policy.shape[0]},total shape -> {policy.shape}\n")
        #print(f"Dirichlet without size parameter has lenght {len(np.random.dirichlet([self.args['dirichlet_alpha']] * (2*self.game.number_of_edges)))}\n")
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                      * np.random.dirichlet([self.args['dirichlet_alpha']] * (2*self.game.number_of_edges), size=policy.shape[0])
        #print(f"Perturbed policy 0: {policy[0]}")
        #print(f"Perturbed policy 1: {policy[1]}")
        #print(f"Policy 1 after shape 0: {policy}")

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]

            valid_moves = spg.game.get_valid_moves()
            spg_policy *= valid_moves
            #print('Masked add_policy: {0}\n'.format(spg_add_policy))

            # Normalize the adapted policy
            spg_policy /= np.sum(spg_policy)
            #print('Normalized Total Perturbed policy: {0}\n'.format(spg_policy))

            # Inizializzazione della root nelle partite parallele
            # non sono sicura che self.game vada bene...
            spg.root = Node(self.game, self.args, states[i], visit_count=1)

            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            print(f"Search = {search}")
            for spg in spGames:
                s =  open(f"{iter}_spgGame_{spg.index}.txt","a")
                j = 1
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    s.write(f"node \n {node.game.graph} at depth {j} is fully expanded\, with {len(node.children)}, visit count = {node.visit_count}\n")
                    node,ucb = node.select(spg.index,iter)
                    j += 1

                s.write(f"At search {search} visiting node at depth {j} with State \n {node.game.graph} \n Ucb = {ucb} \n\n")
                s.close()
                #value, is_terminal = node.game.get_value_and_terminated()
                _, value, win, truncated, _= node.game.step(node.game.extract_action(node.action_taken),simulate=True)
                
                #if is_terminal:
                if win or truncated:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            #print(f"expandable_idx = {expandable_spGames}")

            if len(expandable_spGames) > 0:

                # Ripetere la procedura fatta sopra:
                # costruire lo stack di encoded states da passare alla rete neurale
                #expandable_spGames = spGames[expandable_spGames]
                encoded_states = np.stack([spGames[mappingIdx].game.get_encoded_state() for mappingIdx in expandable_spGames])
                #encoded_states = expandable_spGames[0].game.get_encoded_state()
                
                policy, value = self.model(torch.tensor(encoded_states, device=self.model.device))
                """print(f"Node Policy 0 before perturbation: {policy[0]}")
                if len(expandable_spGames) > 1:
                    print(f"Node Policy 1 before perturbation: {policy[1]}")"""

                policy = torch.softmax(policy, axis=1).cpu().numpy()
                """print(f"Node Policy 0 before perturbation - softmax: {policy[0]}")
                
                if len(expandable_spGames) > 1:
                    print(f"Node Policy 1 before perturbation - softmax: {policy[1]}")"""

                value = value.cpu().numpy()
                """print(f"value 0: {value[0]}")
                if len(expandable_spGames) > 1:
                    print(f"value 1: {value[1]}")"""

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = node.game.get_valid_moves()
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)