import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from flip_environment import FlipEnvironment
from global_environment import GlobalEnvironment
from linear_environment import LinearEnvironment
from local_environment import LocalEnvironment

def select_value_fun():
    def on_select_value_fun(event=None):
        value_fun_name = value_fun_var.get()
        if value_fun_name == "Wagner":
            selected_value_fun = value_fun_wagner
        elif value_fun_name == "Brouwer":
            selected_value_fun = value_fun_brouwer
        else:
            messagebox.showerror("Error", "Select a value function to proceed.")
            return

        root.unbind("<Return>")
        root.destroy()
        select_game(selected_value_fun)

    root = tk.Tk()
    root.title("Select a Conjecture")
    root.geometry("350x250")

    style = ttk.Style()
    #style.theme_use('clam')
    style.theme_use('clam')

    ttk.Label(root, text="Choose a Conjecture:", font=("Helvetica", 16, "bold")).pack(pady=20)

    value_fun_var = tk.StringVar()

    value_funs = ["Wagner", "Brouwer"]
    for fun in value_funs:
        ttk.Radiobutton(root, text=fun, variable=value_fun_var, value=fun).pack(anchor=tk.W, padx=30, pady=5)

    confirm_button = ttk.Button(root, text="Confirm", command=on_select_value_fun)
    confirm_button.pack(pady=20)

    root.bind("<Return>", on_select_value_fun)

    root.mainloop()

def select_game(selected_value_fun):
    def on_select(event=None):
        game_name = game_var.get()
        try:
            number_of_nodes = int(nodes_entry.get())
            if number_of_nodes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of nodes.")
            return

        reward_type = reward_var.get()

        if game_name:
            root.unbind("<Return>")
            nodes_entry.unbind("<Return>")
            root.destroy()
            dense_reward = reward_type == "Incremental"
            main_game(game_name, number_of_nodes, selected_value_fun, dense_reward)

        else:
            messagebox.showerror("Error", "Select a game to proceed.")

    root = tk.Tk()
    root.title("Select a Game")
    root.geometry("400x450")

    style = ttk.Style()
    style.theme_use('clam')

    ttk.Label(root, text="Choose a game:", font=("Helvetica", 16, "bold")).pack(pady=20)

    game_var = tk.StringVar()

    games = ["Flip", "Global", "Linear", "Local"]
    for game in games:
        ttk.Radiobutton(root, text=game, variable=game_var, value=game).pack(anchor=tk.W, padx=30, pady=5)

    ttk.Label(root, text="Enter the number of nodes:", font=("Helvetica", 12)).pack(pady=10)
    nodes_entry = ttk.Entry(root)
    nodes_entry.pack(pady=5)

    ttk.Label(root, text="Select reward type:", font=("Helvetica", 12)).pack(pady=10)
    reward_var = tk.StringVar(value="Sparse")
    reward_types = ["Sparse", "Incremental"]
    for reward in reward_types:
        ttk.Radiobutton(root, text=reward, variable=reward_var, value=reward).pack(anchor=tk.W, padx=30, pady=5)

    confirm_button = ttk.Button(root, text="Confirm", command=on_select)
    confirm_button.pack(pady=20)

    nodes_entry.bind("<Return>", on_select)
    root.bind("<Return>", on_select)

    root.mainloop()

def visualize_state(canvas, figure, graph, current):
    plt.clf()
    pos = nx.circular_layout(graph)
    edge_colors = ['red' if (u == current[0] and v == current[1]) or (u == current[1] and v == current[0]) else 'gray' for u, v in graph.edges()]
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=500, font_size=10)
    u, v = current
    if graph.has_edge(u, v):
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color='red', width=2.0)
    else:
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color='red', style='dashed', width=2.0)
    plt.title("State of the Graph")
    canvas.draw()

def value_fun_wagner(graph, normalize=False):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    if nx.is_connected(g):
        const = 1 + np.sqrt(n - 1)
        radius = max(np.real(nx.adjacency_spectrum(g)))
        weight = len(nx.max_weight_matching(g))
        wagner = const - (radius + weight)
        reward = wagner / n if normalize else wagner
    else:
        reward = -n*2 if normalize else -n*4
    return reward

def value_fun_brouwer(graph, normalize):
    g = nx.Graph(graph)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(g))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1,n+1)])
    diff = sums - (binomials + m)
    reward = max(diff[2:n-2]).real / n if normalize else max(diff[2:n-2]).real
    return reward

def main_game(game_name, number_of_nodes, value_fun, dense_reward):
    if game_name == "Flip":
        game = FlipEnvironment(number_of_nodes=number_of_nodes, value_fun=value_fun, dense_reward=dense_reward)
    elif game_name == "Global":
        game = GlobalEnvironment(number_of_nodes=number_of_nodes, value_fun=value_fun, dense_reward=dense_reward)
    elif game_name == "Linear":
        game = LinearEnvironment(number_of_nodes=number_of_nodes, value_fun=value_fun, dense_reward=dense_reward)
    elif game_name == "Local":
        game = LocalEnvironment(number_of_nodes=number_of_nodes, value_fun=value_fun, dense_reward=dense_reward)

    def perform_action(event=None):
        action = action_entry.get()
        if action.isdigit():
            action = int(action)
            if 0 <= action < game.action_space.n:
                _, reward, done, _, _ = game.step(action)
                graph = nx.from_numpy_array(game.graph)
                status_label.config(text=f"Current position: {game.current}, Reward: {reward:.2f}")
                visualize_state(canvas, figure, graph, game.current)
                if done:
                    messagebox.showinfo("Game Over", "The game is over. The graph is now in a terminal state.")
                    #root.destroy()
            else:
                messagebox.showerror("Invalid Action", "Action out of bounds. Try again.")
        else:
            messagebox.showerror("Invalid Input", "Enter a valid action.")
        action_entry.delete(0, tk.END)

    def exit_game():
        if root:
            root.unbind("<Return>")
            action_entry.unbind("<Return>")
            root.destroy()

    root = tk.Tk()
    root.title(f"Game: {game_name}")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = 800
    window_height = 600

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    action_frame = ttk.Frame(root)
    action_frame.pack(pady=10)

    ttk.Label(action_frame, text="Enter action:", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)
    action_entry = ttk.Entry(action_frame)
    action_entry.pack(side=tk.LEFT, padx=5)
    action_entry.bind("<Return>", perform_action)
    ttk.Button(action_frame, text="Execute", command=perform_action).pack(side=tk.LEFT, padx=5)

    exit_button = ttk.Button(root, text="Exit Game", command=exit_game)
    exit_button.pack(pady=10)

    status_label = ttk.Label(root, text=f"Current position: {game.current}", font=("Helvetica", 12))
    status_label.pack(pady=10)

    figure = plt.figure()
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    graph = nx.from_numpy_array(game.graph)
    visualize_state(canvas, figure, graph, game.current)

    root.protocol("WM_DELETE_WINDOW", exit_game)
    root.mainloop()

if __name__ == "__main__":
    select_value_fun()
