from stable_baselines3.common.callbacks import BaseCallback
import pickle
import os
from graph import Graph

class CheckCallback(BaseCallback):
    """Every check_freq steps, evaluate one episode with the greedy policy and log when a star or a counterexample is found."""
    def __init__(self, eval_env, check_freq, log_file="log.txt", verbose=1):
        super(CheckCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.eval_env = eval_env
        self.log_file = log_file
        self.star_found = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Use the model to interact with the evaluation environment
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, _, done, _, _ = self.eval_env.step(action)

            # Check if the resulting graph is a star
            graph_part = state[:len(state) // 2]
            graph = Graph(graph_part)
            G = graph.graph
            degree_sequence = [d for n, d in G.degree()]
            is_star = degree_sequence.count(1) == len(degree_sequence) - 1 and degree_sequence.count(len(degree_sequence) - 1) == 1

            if is_star:
                with open(self.log_file, 'a') as f:
                    f.write(f"Star found! at training step {self.n_calls}\n")
                self.star_found = True
                return False

            if graph.wagner1() > 0 and graph.is_connected():
                with open(self.log_file, 'a') as f:
                    f.write(f"Counterexample found! at training step {self.n_calls}\n")
                with open(f'counterexample_{self.n_calls}.pkl', 'wb') as f:
                    pickle.dump(graph, f)
                return False

        return True

class CheckOnTrainEnvCallback(BaseCallback):
    """Every check_freq calls to the training env, check the current state and log when a star (if star_check == True) or a counterexample is found."""
    def __init__(self, check_freq, log_file="log.txt", star_check=True, stop_on_star=True, stop_on_counterexample=True, verbose=1):
        super(CheckOnTrainEnvCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_file = log_file
        self.star_check = star_check
        self.stop_on_star = stop_on_star
        self.stop_on_counterexample = stop_on_counterexample
        # self.star_found = False
    
    def _on_training_start(self):
        self.number_of_nodes = self.training_env.get_attr('number_of_nodes')[0]
        self.number_of_edges = self.training_env.get_attr('number_of_edges')[0]

    def _on_step(self) -> bool:
        # Check only if at least number_of_nodes edges have been considered, because otherwise the current graph cannot be connected
        if self.n_calls % self.check_freq == 0 and self.n_calls % self.number_of_edges >= self.number_of_nodes:
            # Get the current state from the training environment
            state = self.training_env.get_attr('state')[0]
            graph_part = state[:len(state) // 2]
            graph = Graph(graph_part)
            G = graph.graph

            # Check if the current state is a star, use this to check if training is working, because the star has wagner1() == 0
            if self.star_check:
                # Compute star condition: one central node of degree number_of_nodes - 1, every oter node of degree 1
                degree_sequence = [d for n, d in G.degree()]
                is_star = degree_sequence.count(1) == len(degree_sequence) - 1 and degree_sequence.count(len(degree_sequence) - 1) == 1

                if is_star:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Star found at env.step() call # {self.n_calls}\n")
                    with open(f'star_{self.n_calls}.pkl', 'wb') as f:
                        pickle.dump(graph, f)
                    # self.star_found = True
                    return not self.stop_on_star

            # Check if the current state is a (connected) counterexample and save it
            if graph.wagner1() > 0 and graph.is_connected():
                with open(self.log_file, 'a') as f:
                    f.write(f"Counterexample found! at training step {self.n_calls}\n")
                with open(f'counterexample_{self.n_calls}.pkl', 'wb') as f:
                    pickle.dump(graph, f)
                return not self.stop_on_counterexample

        return True

def load_results(log_file="log.txt"):
    """
    Load and visualize the graphs from the counterexample files referenced in the log file.

    Parameters:
    log_file (str): Path to the log file.
    """

    # Check if the log file exists
    if not os.path.exists(log_file):
        print("No stars or counterexamples found")
        return

    # Open the log file and read the lines
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Loop over the lines in the log file
    for line in lines:
        # If the line contains 'Counterexample found', load and visualize the corresponding graph
        if 'Counterexample found' in line:
            # Extract the step number from the line
            step = int(line.split()[-1])

            # Construct the pickle file name from the step number
            pickle_file = f'counterexample_{step}.pkl'

            # Load the graph from the pickle file
            with open(pickle_file, 'rb') as f:
                graph = pickle.load(f)

            # Print the Wagner1 score
            print(f"Wagner1 score: {graph.wagner1()}")

            # Draw the graph
            graph.draw()
            plt.show()
