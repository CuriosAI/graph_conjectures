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
        self.num_nodes = self.graph.number_of_nodes()

    def process_nx_graph(self, graph):
        return graph

    def process_linear_graph(self, graph):
        num_nodes = int((-1 + np.sqrt(1 + 8 * len(graph))) / 2)

        # Create an empty graph
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(num_nodes))
        
        # Add edges based on the linear representation
        edge_index = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if graph[edge_index] == 1:
                    G.add_edge(i, j)
                edge_index += 1
        
        return G

    def wagner1(self):
        const = 1 + np.sqrt(self.num_nodes - 1)
        radius = max(np.real(nx.adjacency_spectrum(self.graph)))
        weight = len(nx.max_weight_matching(self.graph))
        return const - (radius + weight)
    
    def is_connected(self):
        return nx.is_connected(self.graph)
    
    def draw(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos)