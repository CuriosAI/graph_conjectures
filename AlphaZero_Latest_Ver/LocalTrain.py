import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
#torch.manual_seed(0)
from tqdm import trange
import networkx as nx

#from Node import Node
#from Graph import Graph 
from local_environment import LocalEnv
#from MCTSParallel import MCTSParallel
from ResNet import ResNet
from AlphaZeroParallel import AlphaZeroParallel
from Alphazero import AlphaZero

#from AlphazeroParallelWagner import AlphaZeroParallel
#from WagnerEnv import WagnerLin
# from AlphazeroWagner import AlphaZero
# from envs import LinEnv
# from ResnetWagner import ResNet
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

#game = LinEnv(18,reward='wagner')
#game = LinEnv(4,reward='wagner')
n = 4
m = int(n*(n-1)/2)
game = LocalEnv(n,wagner_score,2*m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 5,
    'num_iterations': 4,
    'num_selfPlay_iterations': 10,
    'num_parallel_games': 5,
    'num_epochs': 1,
    'batch_size': 10,
    'temperature': 1.25,
    'temperature_iterations': 30,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}
'''
args = {
    'C': 2.5,
    'num_searches': 100,
    'num_iterations': 20,
    'num_selfPlay_iterations': 500,
    #'num_parallel_games': 500,
    'num_epochs': 5,
    'batch_size': 512,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.05,
    'dirichlet_alpha': 0.3
}'''

alphaZero = AlphaZero(model, optimizer, game, args)
#alphaZero = AlphaZeroParallel(model, optimizer, game, args)
# training_start
start_time = time.time()
alphaZero.learn()
end_time = time.time()
execution_time = end_time - start_time
current_time = datetime.datetime.now()
with open(f"Local{current_time.day}_{current_time.hour}_{current_time.minute}_execution_time_{game.number_of_nodes}.txt", "w") as file:  # Usa "a" per appendere o "w" per sovrascrivere
    file.write(f"Execution_time: {execution_time} s \n")
    file.write(f"Number_of_nodes : {game.number_of_nodes} \n")
    file.write(f"C = {args['C']} \n")
    file.write(f"num_searches = {args['num_searches']} \n")
    file.write(f"num_iterations = {args['num_iterations']} \n")
    file.write(f"num_selfPlay_iterations = {args['num_selfPlay_iterations']} \n")
    file.write(f"num_parallel_games = {args['num_parallel_games']} \n")
    file.write(f"num_epochs = {args['num_epochs']} \n")
    file.write(f"batch_size = {args['batch_size']} \n")
    file.write(f"temperature = {args['temperature']} \n")
    file.write(f"dirichlet_epsilon = {args['dirichlet_epsilon']} \n")
    file.write(f"dirichlet_alpha = {args['dirichlet_alpha']} \n")
# training_end
#training_time
# utilizzo gpu ----> utile
# chiedere ad Andrea per parametri tecnici da specificare 
# per le performance del software
# max e min dei core gpu
