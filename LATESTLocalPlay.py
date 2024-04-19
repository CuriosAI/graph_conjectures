import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
#torch.manual_seed(0)
from tqdm import trange
from Node import Node
#from Graph import Graph
from local_environment import LocalEnv
from MCTS import MCTS
from AlphaZeroParallel import AlphaZeroParallel
from ResNet import ResNet

import time
import datetime

#game = LinEnv(4,reward='wagner',normalize_reward=False)
def wagner_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    if nx.is_connected(g):
        info = {'connected'}
        const = 1 + np.sqrt(n - 1)
        radius = max(np.real(nx.adjacency_spectrum(g)))
        weight = len(nx.max_weight_matching(g))
        wagner = const - (radius + weight)
        reward = wagner / n if normalize else wagner
    else:
        reward = -n*2 if normalize else -n*4  # penalty to be used when the conjecture
        # holds only for connected graphs.
        # This normalization assumes that other rewards are > -1
        # we use info to pass the terminal state
        # (Flora) : info può essere utilizzato x mantenere l'informazione sulla connessione
        # in modo da non doverla ricontrollare (modifica in sospseso)
        info = {'not_connected'}
    
    return reward, info

n = 4
stop = 4*int(n*(n-1)/2)
game = LocalEnv(n,wagner_score,stop,normalize_reward=False)

args = {
    # Exploration parameter for the MCTS algorithm
    'C': 2,
    # Number of MCTS simulations per move
    'num_searches': 200,
    # Dirichlet noise for selecting initial exploration moves
    'dirichlet_epsilon': 0.05,
    # Dirichlet noise parameter
    'dirichlet_alpha': 0.3
}
# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the neural network model
model = ResNet(game, 9, 128, device)
# Load the trained model parameters
''' Allenati con AlphaZero '''
#model.load_state_dict(torch.load("model_0.pt", map_location=device))
model.load_state_dict(torch.load("model_0_4.pt", map_location=device))
# model.load_state_dict(torch.load(f"model_{number_of_nodes}.pt", map_location=device))
# Set the model to evaluation mode
model.eval()
''' Quì non c'è nessun aggiornamento
Perché, allora, usiamo ancora l'esplorazione di MCTS? '''

# Create an instance of the MCTS algorithm
mcts = MCTS(game, args, model)
# Start the game with an empty board
'''Occhio: questo stato iniziale non serve solo per il print,
va anche passato ad mcts
Giulia
state = game.get_initial_state()'''
# ritorna la matrice triangolare superiore
                                 # reset
state, _ = game.reset()

current_time = datetime.datetime.now()
play = open(f"LocalMatch_on_{game.number_of_nodes}_{current_time.day}_{current_time.hour}_{current_time.minute}.txt", "w")
play = open(f"LocalMatch_on_{game.number_of_nodes}_{current_time.day}_{current_time.hour}_{current_time.minute}.txt", "a")
play.write(f"MATCH ON {game.number_of_nodes} NODES WITH LOCALENV. Stop after {stop} rounds\n")
if game.start_with_complete_graph:
    play.write("Starting from complete graph\n")
else:
    play.write("Starting from empty graph\n")

print("Start to play:")
start_play = time.time()
while True:
    print(state)
    print(f"Plain score = {game.old_value}")

    mcts_probs = mcts.search(state)
    print(f"MCTS probs : {mcts_probs}")
    # Selects the move based on the highest probability
    action = game.extract_action(np.argmax(mcts_probs))
    print(f"action: {action}")
    play.write(f"Round {game.timestep_it} - Position {game.position} - Action {action} --> ")
    if action >= game.number_of_nodes:
        play.write(f"Flip edge ({game.position},{action-game.number_of_nodes})\n")
    else:
        play.write(f"Leave unchanged edge ({game.position},{action})\n")
    # Update the game state with the selected move

    state, reward, done, truncated, info = game.step(action)
    play.write(f"Obtained reward = {reward}\n")
    
    # Checks the game status and win
    if done or truncated:
        print(f"Is terminal\n")
        # Prints the final board
        print(game.graph)
        print(f"Score = {reward}")
        
        play.write(f"Playing time : {time.time()-start_play} s\n")
        play.write(f"Last score = {reward}\n")
        if done:
            print("Counterexample found")
        else:
            print("Counterexample not found")
        break

play.close()

#graph = Graph(state,True).graph
nx.draw(nx.Graph(game.graph), with_labels = True)
plt.show()