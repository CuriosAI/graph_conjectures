import numpy as np
import random
import math
#print(np.__version__)
import torch
import copy
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
# torch.manual_seed(0)
#from tqdm.notebook import trange
from tqdm import trange
import time
import datetime

from MCTSParallel import MCTSParallel
#from Node import Node

class SPG:
    def __init__(self, game, idx):
        # self.state = game.get_initial_state() ''' reset '''
        self.state, _ = game.reset()
        self.game = copy.deepcopy(game)
        self.memory = []
        self.root = None # radice locale, nodo padre di dove sono
        self.node = None # dove sono
        self.index = idx
    
class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args): 
        """
        Initialize the AlphaZero agent with a neural network model, optimizer, game environment, and hyperparameters.

        Args:
            model: The neural network model to be used for policy and value estimation.
            optimizer: The optimizer to update the model weights.
            game: The game environment to play in.
            args: A dictionary containing hyperparameters for the algorithm.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game 
        self.args = args
        self.mcts = MCTSParallel(game, args, model) 
        self.value_losses = []  # List to store value losses
        self.policy_losses = []
        self.game_idx = 0
        self.selfplayiter = 0

    def selfPlay(self):
        """
        Perform self-play games and collect data for training.

        Returns:
            A list of transitions (encoded state, policy probabilities, outcome) collected during self-play.
        """
        return_memory = []
        
        # quì vengono create le partite parallele
        # partono tutte dallo stesso stato iniziale (lo stato corrente)
        # ogni elemento è un SPG con gioco LinEnv
        spGames = [SPG(self.game,spg) for spg in range(self.args['num_parallel_games'])]
        for spg in spGames:
            f = open(f"{self.selfplayiter}_spgGame_{spg.index}.txt","w")
            f.close()
            #f = open(f"{self.selfplayiter}_ucb_values{spg.index}.txt","w")
            #f.close()

        while len(spGames) > 0:
            #print(f"Size of neutral_states : {len(spGames)}")
            neutral_states = np.stack([spg.state for spg in spGames]) 
            #print(f"Size of neutral_states : {neutral_states.shape}")
            # All'inizio, questi stati sono tutti uguali
            # lo saranno anche dopo?

            self.mcts.search(neutral_states, spGames, self.selfplayiter) # <- spGames viene passato ad MCTSParallel
                                                      # il costruttore di MCTSParallel prende in input
                                                      # game, args, model
                                                      # è viene inizializzato come parametro di AlphaZero
                                                      # pertanto, nel game di mcts c'è lo stesso game con cui viene inizializzato
                                                      # AlphaZero (all'inizio sarà lo stato iniziale).
                                                      # Come avanzerà?
            # DOMANDA: i neutral states potrebbero essere eliminati?
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                # action_probs = np.zeros(self.game.action_size) 
                action_probs = np.zeros(2*self.game.number_of_edges) 
                # dipende dalla modalità di gioco
                # in LinEnv action_size = 2
                # (nella versione semplificata)
                
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs))

                if self.game_idx < self.args['temperature_iterations']:
                    temp = self.args['temperature']
                else:
                    temp = 1
                temperature_action_probs = action_probs ** (1 / temp)

                #action = np.random.choice(self.game.action_size, p=temperature_action_probs/np.sum(temperature_action_probs))
                action = np.random.choice(2*self.game.number_of_edges, p=temperature_action_probs/np.sum(temperature_action_probs))

                #spg.state = self.game.get_next_state(spg.state, action, spg.player)
                _, value, win, truncated, _ = spg.game.step(action)
                # value, is_terminal = spg.game.get_value_and_terminated()

                if win or truncated:
                    print("is_terminal\n")
                    for hist_neutral_state, hist_action_probs in spg.memory:
                        hist_neutral_state = spg.game.get_encoded_state()
                        hist_outcome = value #if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            hist_neutral_state,
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
        
        self.game_idx += 1

            #player = self.game.get_opponent(player)

        return return_memory

    # INDIP DA ENV (?)
    def train(self, memory):
        # NB : Abbiamo aggiunto il player
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            print(f"Batch ={batchIdx}")
            #sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            print(f"Sample size ={len(sample)}")
            state, policy_targets , value_targets= zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            print(f"State size ={state.size()}")
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # INDIP DA ENV
    def save_losses(self, filename_prefix):
        torch.save({'policy_losses': self.policy_losses, 'value_losses': self.value_losses}, f"{filename_prefix}_losses.pt")

    # INDIP DA ENV
    def learn(self):
        # OSS: num_iter != num_epochs!
        # per ogni giro di learning, ci vengono fatte 5 epoche di training
        # in totale ottengo 25 epoche di training da eseguire
        # solo che, ogni blocco di 5 epoche elabora un training set diverso
        # man mano che i giri di learning avanzano, il training set che ottengo è più orientato
        # verso le partite buone?
        current_time = datetime.datetime.now()
        f = open(f"{current_time.day}_{current_time.hour}_{current_time.minute}_time_for_iterations.txt","w")
        f = open(f"{current_time.day}_{current_time.hour}_{current_time.minute}_time_for_iterations.txt","a")
        for iteration in range(self.args['num_iterations']):
            memory = []
            print(f"Iteration ={iteration}")

            self.model.eval() # NON AGGIORNIAMO I PESI
            start_iter = time.time()
            # trange() è range ma con la barra del progresso
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                print(f"Self_play_iteration = {selfPlay_iteration}")
                memory += self.selfPlay()
                # appendi un qualche esito
                # con questi parametri, memory avrà due blocchi
            f.write(f"Self Playing time for iter {iteration} : {time.time()-start_iter} s \n")
            self.selfplayiter += 1

            start_train = time.time()
            # quì alphazero usa le informazioni
            # ricavate dalle 1000 partite giocate per aggiornarsi
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                print(f"Epoch = {epoch}")
                self.train(memory)
            f.write(f"Training time for iter {iteration} : {time.time()-start_train} s \n")
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game.number_of_nodes}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game.number_of_nodes}.pt")
            self.save_losses(f"model_{iteration}_{self.game.number_of_nodes}")
        
        f.close()
            

# spg = self play game
# partite fatte da alphazero
# la radice sarà lo stato di partenza ?
# e dove sono i children?