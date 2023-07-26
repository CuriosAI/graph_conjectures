import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import pickle
import glob
import re
import networkx as nx

from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from envs import LinEnv
from graph import Graph

def create_experiment_folder(algo, number_of_nodes):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = now.strftime("%Y%m%d%H%M%S")

    # Use the date and time string to create a unique folder name
    unique_folder = f"experiments/{date_time_string}-{algo}_{number_of_nodes}"

    return unique_folder

def parse_unique_folder(unique_folder):
    # Split the unique folder string into parts
    parts = unique_folder.split("-")

    # The algorithm name and number of nodes are in the last part
    algo_and_nodes = parts[-1].split("_")

    # Extract the algorithm name and number of nodes
    algo = algo_and_nodes[0]
    number_of_nodes = int(algo_and_nodes[1])

    return algo, number_of_nodes

# def print_layer_dims(model):
#     policy_net_dims = [layer.out_features for layer in model.policy.mlp_extractor.policy_net.children() if isinstance(layer, torch.nn.Linear)]
#     value_net_dims = [layer.out_features for layer in model.policy.mlp_extractor.value_net.children() if isinstance(layer, torch.nn.Linear)]

#     print(f"policy_net = {policy_net_dims}, value_net = {value_net_dims}")

def create_graphs(model, eval_env, number_of_edges, deterministic):
    state, _ = eval_env.reset()
    action_probs = []
    for _ in range(number_of_edges):
        action, state = model.predict(state, deterministic=deterministic)
        if state is not None and 'action_proba' in state:
            action_probs.append(state['action_proba'])
        state, _, done, _, _ = eval_env.step(action)
        if done:
            graph = Graph(state[:number_of_edges])
    return graph, action_probs

def read_experiment(unique_folder):
    """Read eval_callback and best_model folders, and visualize one greedy and 5 non-greedy graphs from the best_model policy. For DQN, the non-greedy graphs are produced by epsilong-greedy policy, with the default epsilon = 0.05"""

    eval_data = f"{unique_folder}/eval_callback/evaluations.npz"
    best_model = f"{unique_folder}/best_model/best_model.zip"
    algo, number_of_nodes = parse_unique_folder(unique_folder)

    ######################
    # Show evaluation data
    ######################

    data = np.load(eval_data)

    # Create a dictionary to hold the data
    data_dict = {}

    # Process each array in the data
    for key in data.keys():
        # Flatten the array if it's 2-dimensional
        if data[key].ndim == 2 and data[key].shape[1] == 1:
            data_dict[key] = data[key].flatten()
        else:
            # Print a warning if the array is not 1-dimensional
            if data[key].ndim != 1:
                print(f"Warning: The array '{key}' is not 1-dimensional.")
            data_dict[key] = data[key]

    # Create and print the DataFrame
    df = pd.DataFrame(data_dict)
    print(df)

    # Find the index of the maximum result and print the corresponding line
    max_result_index = df['results'].idxmax()
    print(df.loc[max_result_index])

    # Load best model
    if algo == 'DQN':
        model = DQN.load(best_model)
    elif algo == 'PPO':
        model = PPO.load(best_model)
    else:
        print(f"Model type {algo} not recognized.")

    eval_env = LinEnv(number_of_nodes=number_of_nodes, normalize_reward=False)
    number_of_edges = eval_env.number_of_edges # This is the horizon for LinEnv, should be changed for different envs

    # # Print layers dimensions
    # print_layer_dims(model)
    # # layers = model.policy.mlp_extractor
    # # layer_dims = [layer.out_features for layer in layers.modules() if hasattr(layer, 'out_features')]
    # # print(layer_dims)
    # # print(layers)

    # Create the "optimal" graph
    optimal_graph, optimal_action_probs = create_graphs(model, eval_env, number_of_edges, deterministic=True)

    # Create 5 graphs sampled by the optimal policy. Useless for DQN, because actions are epsilon-random
    sampled_graphs = []
    sampled_action_probs = []
    for _ in range(5):
        graph, action_probs = create_graphs(model, eval_env, number_of_edges, deterministic=False)
        sampled_graphs.append(graph)
        sampled_action_probs.append(action_probs)

    # Set the default font size
    plt.rc('font', size=7)

    # Create a single plot with all the graphs
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))  # 3 rows, 2 columns

    # Plot the "optimal" graph
    optimal_graph.draw(ax=axs[0, 0])
    if isinstance(model, DQN):
        axs[0, 0].set_title(f"Greedy optimal graph (DQN)", y=0.8, x=0.15)
    else:
        axs[0, 0].set_title(f"Greedy optimal graph (PPO)\nAction probabilities: {optimal_action_probs}", y=0.8, x=0.15)
    # Add a border around the subplot
    axs[0, 0].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', lw=2, transform=axs[0, 0].transAxes, clip_on=False))

    # Plot the sampled graphs
    for i in range(5):
        row = (i + 1) // 2
        col = (i + 1) % 2
        sampled_graphs[i].draw(ax=axs[row, col])
        if isinstance(model, DQN):
            axs[row, col].set_title(f"Sampled graph {i+1} (DQN)", y=0.8, x=0.15)
        else:
            axs[row, col].set_title(f"Sampled graph {i+1} (PPO)\nAction probabilities: {sampled_action_probs[i]}", y=0.8, x=0.15)
        # Add a border around the subplot
        axs[row, col].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', lw=2, transform=axs[row, col].transAxes, clip_on=False))

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Reduce the top margin to prevent the top graphs from being cut off
    plt.show()

def show_counterexamples(unique_folder):
    """
    Load and visualize the graphs from the counterexample and star files in the folder.

    Parameters:
    unique_folder (str): Path to the counterexamples folder.
    """

    counterexamples_folder = f"{unique_folder}/counterexamples"
    
    # Get the list of all pickle files in the counterexamples folder
    pickle_files = glob.glob(f"{counterexamples_folder}/*.pkl")

    # List to keep track of the graphs we've already seen
    seen_graphs = []

    # Counter for the number of skipped graphs
    skipped_graphs = 0

    # Loop over the pickle files
    for pickle_file in pickle_files:
        # Extract the step number from the file name
        match = re.search(r'(counterexample|star)_(\d+)_(\d+).pkl', pickle_file)
        graph_type = match.group(1)
        step = int(match.group(3))

        # Load the graph from the pickle file
        with open(pickle_file, 'rb') as f:
            graph = pickle.load(f)

        # Check if the graph is isomorphic to any of the previous ones
        if any(nx.is_isomorphic(graph.graph, g.graph) for g in seen_graphs):
            skipped_graphs += 1  # Increment the counter
            continue  # Skip this graph

        # Add the graph to the list of seen graphs
        seen_graphs.append(graph)

        # Create the title string
        title = f"{graph_type} found at step: {step}"

        # Draw the graph with the title
        graph.draw(title=title)
        plt.show()

    # Print the number of skipped graphs
    print(f"{skipped_graphs} graphs skipped, because isomorphic to one of the graphs shown")


def make_env(env_id):
    def _init():
        env = LinEnv(env_id)
        return env
    return _init

def make_normalized_linenv(number_of_nodes):
    def _init():
        env = LinEnv(number_of_nodes=number_of_nodes, normalize_reward=True)
        return env
    return _init