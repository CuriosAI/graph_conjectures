import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Graph:
    def __init__(self, graph):
        if isinstance(graph, nx.Graph):
            self.graph = self.process_nx_graph(graph)
        elif isinstance(graph, np.ndarray):
            self.graph = self.process_linear_graph(graph)
        else:
            raise TypeError("graph must be a networkx Graph or a linear graph")
        self.number_of_nodes = self.graph.number_of_nodes()

    def process_nx_graph(self, graph):
        return graph

    def process_linear_graph(self, graph):
        """A linear graph is a (non-directed, unlabeled, without loops) graph described by a ndarray of shape (edges, ), where edges (i,j) are in lessicografic order"""
        assert all(element in [0, 1] for element in graph), "graph is a ndarray, but it contains elements other than 0 and 1"
        # nodes_if_graph_is_without_timestep = int((1 + np.sqrt(1 + 8 * len(graph))) / 2)
        # nodes_if_graph_is_with_timestep = int((1 + np.sqrt(1 + 4 * len(graph))) / 2)
        # condition_without = (nodes_if_graph_is_without_timestep == (1 + np.sqrt(1 + 8 * len(graph))) / 2)
        # condition_with = (nodes_if_graph_is_with_timestep == (1 + np.sqrt(1 + 4 * len(graph))) / 2)
        # assert condition_without ^ condition_with, f"wrong len(graph): one and only one between (1 + np.sqrt(1 + 8 * len(graph))) / 2 = {(1 + np.sqrt(1 + 8 * len(graph))) / 2} and (1 + np.sqrt(1 + 4 * len(graph))) / 2 = {(1 + np.sqrt(1 + 4 * len(graph))) / 2} should be integer"
        # if condition_without:
        #     number_of_nodes = nodes_if_graph_is_without_timestep
        # elif condition_with:
        #     number_of_nodes = nodes_if_graph_is_with_timestep
        #     edges = int(len(graph) / 2)
        #     graph = graph[:edges] # remove the timestep part
        # else:
        #     exit("this point in the software should not be reachable, what is happening?")
        number_of_nodes = int((1 + np.sqrt(1 + 8 * len(graph))) / 2)
        condition = (number_of_nodes == (1 + np.sqrt(1 + 8 * len(graph))) / 2)
        assert condition, f"wrong len(graph): 'number_of_nodes' = (1 + np.sqrt(1 + 8 * len(graph))) / 2 = {(1 + np.sqrt(1 + 8 * len(graph))) / 2} should be integer"

        # Create an empty graph
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(number_of_nodes))

        # Add edges ordered by the order of the nodes
        edge_index = 0
        for i in range(number_of_nodes):
            for j in range(i + 1, number_of_nodes):
                if graph[edge_index] == 1:
                    G.add_edge(i, j)
                edge_index += 1

        return G

    def wagner1(self):
        const = 1 + np.sqrt(self.number_of_nodes - 1)
        radius = max(np.real(nx.adjacency_spectrum(self.graph)))
        weight = len(nx.max_weight_matching(self.graph))
        return const - (radius + weight)

    def is_connected(self):
        return nx.is_connected(self.graph)

    def is_star(self):
        # Compute star condition: one central node of degree number_of_nodes - 1, every other node of degree 1
        degree_sequence = [d for n, d in self.graph.degree()]
        is_star = degree_sequence.count(1) == len(degree_sequence) - 1 and degree_sequence.count(len(degree_sequence) - 1) == 1
        return is_star

    def draw(self):
        # Create a figure and axes
        fig, ax = plt.subplots()
        pos = nx.spring_layout(self.graph)
        ax.clear()
        # Draw the new graph
        #plt.title(f"wagner1 score = {graph.wagner1()} = {const} - ({radius} + {weight}) = sqrt(8) + 1 - (radius + weight)")
        nx.draw(self.graph, pos=pos, ax=ax, with_labels=True)
        # Update the display
        plt.draw()
        # Pause for a moment to show the plot
        plt.pause(1)
        # Keep the window open
        #plt.show()
