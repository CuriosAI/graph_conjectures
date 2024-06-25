import numpy as np

print(np.__version__)
import torch
print(torch.__version__)
import matplotlib.pyplot as plt
#torch.manual_seed(0)
import networkx as nx

# OSS: MCTS normale, non parallelo
# from WagnerEnv import WagnerLin
from MCTSWagner import MCTS
from envs import LinEnv
from ResnetWagner import ResNet

""" This code implements a Reinforcement Learning algorithm to test the First Wagner's conjecture. 
It uses a neural network-based model to evaluate the state of the game and an AlphaZero-like Monte-Carlo Tree Search (MCTS) algorithm to make decisions."""

# Create an instance of the WagnerLin game class
# normalize_reward = True during training only
# game = LinEnv(18,reward='wagner',normalize_reward=False)
game = LinEnv(5,reward='wagner',normalize_reward=False)

# Set the player to 0
player = 0
# ?? Serve? Non chiaro, probabilmente no

args = {
    # Exploration parameter for the MCTS algorithm
    'C': 2,
    # Number of MCTS simulations per move
    'num_searches': 10,
    # Dirichlet noise for selecting initial exploration moves
    'dirichlet_epsilon': 0.25,
    # Dirichlet noise parameter
    'dirichlet_alpha': 0.3
}
# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the neural network model
model = ResNet(game, 6, 64, device)
# Load the trained model parameters
''' Allenati con AlphaZero '''
#model.load_state_dict(torch.load("model_0.pt", map_location=device))
model.load_state_dict(torch.load("model_4_WagnerLin_18.pt", map_location=device))
# model.load_state_dict(torch.load(f"model_{number_of_nodes}.pt", map_location=device))
# Set the model to evaluation mode
model.eval()
''' Quì non c'è nessun aggiornamento
Perché, allora, usiamo ancora l'esplorazione di MCTS? '''

# Create an instance of the MCTS algorithm
mcts = MCTS(game, args, model) ''' <-- ! A cosa serve l'MCTS quì? Non basta la ResNet?
                                         Credevo che MCTS servisse solo durante l'exec
                                         di AlphaZero per apprendere i pesi, e che per
                                         giocare bastasse la rete... '''
# Start the game with an empty board
'''Occhio: questo stato iniziale non serve solo per il print,
va anche passato ad mcts
Giulia
state = game.get_initial_state()'''
# ritorna la matrice triangolare superiore
                                 # reset
state = game.reset()
'''
adjacency_dict = {0: (1,2,3,4), 1: (0,), 2: (0,), 3:(0,), 4:(0,) }
G = nx.from_dict_of_lists(adjacency_dict)
state = nx.adjacency_matrix(G).todense()
state = np.triu(state)
print(state)
print(game.wagner_score(state))
graph = nx.from_numpy_array(state)
nx.draw(graph, with_labels = True)
plt.show()
'''
while True:
    print(state)
    #graph = nx.from_numpy_array(state)
    #nx.draw(graph, with_labels = True)
    #plt.show()
    
    print(f"Plain score = {game.score()[0]}")

    mcts_probs = mcts.search(state)
    # Selects the move based on the highest probability
    ''' La policy finale è MCTS policy ?! '''

    encoded_action = np.argmax(mcts_probs)
    action = game.extract_action(encoded_action)
    # Update the game state with the selected move
    # copy.deepcopy(self.state), reward, self.done, False, info
    state, reward, terminated, _, info = game.step(action)
    # Checks the game status and winner
    value, is_terminal = game.get_value_and_terminated()
    
    if is_terminal:
        # Prints the final board
        print(state)
        print(game.score()[0])
        if value == 1:
            print("Counterexample found")
        else:
            print("Counterexample not found")
        break


graph = nx.from_numpy_array(state)
nx.draw(graph, with_labels = True)
plt.show()