import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.linear_environment import LinearEnvironment
from envs.local_environment import LocalEnvironment
from envs.global_environment import GlobalEnvironment

import os
import csv

# ----------------- REWARD FUNCTIONS -----------------

def brouwer_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1, n+1)])
    diff = sums - (binomials + m)
    if normalize:
        return np.tanh(max(diff[2:n-2]))
    else:
        return max(diff[2:n-2])

def central_brouwer_score(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1, n+1)])
    diff = sums - (binomials + m)
    lower = int(0.25 * n)
    upper = int(0.75 * n) - 1
    if normalize:
        return np.tanh(max(diff[lower:upper]))
    else:
        return max(diff[lower:upper])

def compute_reward_stats(graph, mode="complete"):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1, n+1)])
    diff = sums - (binomials + m)

    if mode == "central":
        lower = int(0.25 * n)
        upper = int(0.75 * n) - 1
        max_index = np.argmax(diff[lower:upper]) + lower
    else:
        max_index = np.argmax(diff[2:n-2]) + 2

    return m, max_index

# ----------------- UTILS -----------------

def get_action_mask(env):
    return np.array(env.unwrapped.get_valid_actions(), dtype=np.int32)

def save_result_to_csv(path, row, header=None):
    file_exists = os.path.isfile(path)
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        writer.writerow(row)

def compute_graph_properties(G):
    props = {}
    props["density"] = nx.density(G)
    props["avg_degree"] = np.mean([d for _, d in G.degree()])
    props["max_degree"] = max(dict(G.degree()).values())
    props["avg_clustering"] = nx.average_clustering(G)
    props["num_components"] = nx.number_connected_components(G)

    if nx.is_connected(G):
        props["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        props["avg_shortest_path"] = None

    return props

def draw_graph_and_save(graph, reward_history, env_name, number_of_nodes, model_mode, score, path_dir):
    fig, (ax_graph, ax_reward) = plt.subplots(1, 2, figsize=(14, 6))

    g = nx.Graph(graph)
    pos = nx.circular_layout(g)
    nx.draw(g, pos=pos, ax=ax_graph, with_labels=True,
            node_color="skyblue", edge_color="black",
            width=1, node_size=500, font_size=10)
    if model_mode == "complete":
        title = f"Environment {env_name} | {number_of_nodes} Nodes | Complete Reward"
    elif model_mode == "central":
        title = f"Environment {env_name} | {number_of_nodes} Nodes | Central Reward"
    ax_graph.set_title(title, fontsize=16, fontweight="bold")

    ax_graph.text(
        0.5, -0.065, f"Final score: {score}",
        fontsize=16, ha='center', transform=ax_graph.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="black", linewidth=1)
    )

    ax_reward.plot(reward_history, marker="o", color="green")
    ax_reward.set_title("Score Evolution", fontsize=16, fontweight="bold")
    ax_reward.set_xlabel("Steps")
    ax_reward.set_ylabel("Score")
    ax_reward.grid(alpha=0.4)

    plt.tight_layout()
    os.makedirs(path_dir, exist_ok=True)
    path = f"{path_dir}/final_graph_{env_name}_{number_of_nodes}_{model_mode}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# ----------------- MAIN TEST FUNCTION -----------------

def run_test(env_class, env_name, number_of_nodes, model_mode, num_games=5):
    reward_fn = central_brouwer_score if model_mode == "central" else brouwer_score
    model_path = f"./models/{env_name}/ppo_{number_of_nodes}_{model_mode}"
    plot_path = f"./plots"
    csv_path = f"./results/games_metrics_{num_games}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    model = MaskablePPO.load(model_path)

    best_score = -np.inf
    best_graph = None
    best_history = None
    best_game_id = -1

    for game_id in range(num_games):
        game = env_class(number_of_nodes, reward_fn, normalize_reward=False)
        game = ActionMasker(game, get_action_mask)
        state, _ = game.unwrapped.reset()
        initial_score = reward_fn(game.unwrapped.graph, normalize=False)
        reward_history = [initial_score]

        while True:
            action_mask = get_action_mask(game)
            action, _ = model.predict(state, deterministic=False, action_masks=action_mask)
            state, reward, done, truncated, info = game.unwrapped.step(action)
            G = nx.Graph(game.unwrapped.graph)
            score = reward_fn(G, normalize=False)
            num_edges, t_index = compute_reward_stats(G, mode=model_mode)
            props = compute_graph_properties(G)
            num_edges, t_index = compute_reward_stats(game.unwrapped.graph, mode=model_mode)
            reward_history.append(score)

            if done or truncated:
                # Save to CSV
                header = ["Env", "Size", "Reward type", "Game id", "Final reward", "Num edges", "t* index",
                "density", "avg_degree", "max_degree", "avg_clustering", "num_components", "avg_shortest_path"]

                row = [
                    env_name, number_of_nodes, model_mode, game_id, score, num_edges, t_index,
                    props["density"], props["avg_degree"], props["max_degree"],
                    props["avg_clustering"], props["num_components"], props["avg_shortest_path"]
                ]

                save_result_to_csv(
                    path=csv_path,
                    row=row,
                    header=header
                )
                if score > best_score:
                    best_score = score
                    best_graph = game.unwrapped.graph.copy()
                    best_history = reward_history.copy()
                    best_game_id = game_id
                break

    draw_graph_and_save(
        best_graph,
        best_history,
        env_name,
        number_of_nodes,
        model_mode,
        best_score,
        plot_path
    )

# ----------------- LOOP OVER CONFIGURATIONS -----------------

env_classes = {
    "Linear": LinearEnvironment,
    "Local": LocalEnvironment,
    "Global": GlobalEnvironment
}
sizes = [11, 12, 13, 14, 15]
modes = ["complete", "central"]
num_games = 50
for env_name, env_class in env_classes.items():
    for number_of_nodes in sizes:
        for model_mode in modes:
            try:
                print(f"\n▶ Running {env_name} | Size {number_of_nodes} | Mode: {model_mode}")
                run_test(env_class, env_name, number_of_nodes, model_mode, num_games=num_games)
            except Exception as e:
                print(f"❌ Failed for {env_name} | {number_of_nodes} | {model_mode}: {e}")
