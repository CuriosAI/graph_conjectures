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
#from linear_environment import LinEnv
from linear_environment import LinEnv
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
    return reward,info

def brouwer_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = n*(n-1)//2
    lamb = np.flip(nx.laplacian_spectrum(g)) # <-- pure questo costa...bisognerebbe ridurre!!!
                                             # studiare meglio la congettura per ridurre
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1,n+1)])
    diff = sums - (binomials + m)
    if normalize:
        return max(diff[2:n-2])/n, {}
    else:
        return max(diff[2:n-2]), {}
    
#game = LinEnv(4,wagner_score,normalize_reward=False)
game = LinEnv(7,wagner_score,normalize_reward=False)

args = {
    # Exploration parameter for the MCTS algorithm
    'C': 2,
    # Number of MCTS simulations per move
    'num_searches': 25,
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
for i in range(20):
    #model.load_state_dict(torch.load("model_0.pt", map_location=device))
    model.load_state_dict(torch.load(f"model_{i}_7.pt", map_location=device))
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
    play = open(f"{i}LinearMatch_on_{game.number_of_nodes}_{current_time.day}_{current_time.hour}_{current_time.minute}.txt", "w")
    play = open(f"{i}LinearMatch_on_{game.number_of_nodes}_{current_time.day}_{current_time.hour}_{current_time.minute}.txt", "a")
    play.write(f"MATCH ON {game.number_of_nodes} NODES WITH LINENV\n")
    if game.start_with_complete_graph:
        play.write("Starting from complete graph\n")
    else:
        play.write("Starting from empty graph\n")

    print("Start to play:")
    start_play = time.time()
    k = 0
    i = 0
    if game.self_loops:
        j = 0
    else:
        j = 1

    while True:
        print(state)
        print(f"Plain score = {game.old_value}")

        mcts_probs = mcts.search(state)
        print(f"MCTS probs : {mcts_probs}")
        # Selects the move based on the highest probability

        action = game.extract_action(np.argmax(mcts_probs))
        print(f"action: {action}")

        play.write(f"Round {k} - Action {action} --> ")
        if action == 1:
            play.write(f"Flip edge ({i},{j})\n")
        else:
            play.write(f"Leave unchanged edge ({i},{j})\n")

        # Update the game state with the selected move
        # copy.deepcopy(self.state), reward, self.done, False, info
        state, reward, done, truncated, info = game.step(action)
        k += 1
        j += 1
        if j >= game.number_of_nodes:
            i += 1
            if game.self_loops:
                j = i
            else:
                j = i+1
            # non metto questa condizione: non credo che ci saranno problemi
            #if i > game.number_of_nodes:
                
        
        # Checks the game status and winner
        # value, is_terminal = game.get_value_and_terminated()
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
    #graph = Graph(state,True).graph
    g = open(f"Linear_FinalGraph_{current_time.day}_{current_time.hour}_{current_time.minute}.g6", "w")
    g = open(f"Linear_FinalGraph_{current_time.day}_{current_time.hour}_{current_time.minute}.g6", "w")
    #ipazia
    # g.write(nx.to_graph6_bytes(nx.Graph(game.graph),header=False))
    #compute-clai
    g.write(nx.to_graph6_bytes(nx.Graph(game.graph),header=False).decode('utf-8'))
    g.close()
    nx.draw(nx.Graph(game.graph), with_labels = True)
    plt.show()
    plt.savefig(f"{i}Linear_FinalGraph")