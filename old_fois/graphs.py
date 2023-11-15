import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
g = nx.Graph()
n = 19
g.add_nodes_from(range(n))
g.add_edges_from([(0,17),(0,2),(0,7),(0,10),(0,15),(0,14),(0,12),(16,3),(0,16),(16,1),(1,11),(1,4),(1,8),(1,9),(1,5),(1,13),(1,6),(1,18)])
# for i in range(n):
#     g.add_edge(i,i)
# g.add_nodes_from(range(n))
# g.add_edges_from([(0,2),(0,7),(0,10),(0,14),(0,12),(0,3),(0,15),(15,1),(1,11),(1,4),(1,8),(1,9),(1,5),(1,13),(1,6)])
mu = len(nx.max_weight_matching(g, maxcardinality=True))
print(nx.adj_matrix(g))
lambda_1 = max(nx.adjacency_spectrum(g))
print(mu)
print(lambda_1)
print(mu + lambda_1)
print(np.sqrt(n-1)+1)
print(mu + lambda_1 - (np.sqrt(n-1)+1))
plt.figure(1)
nx.draw_kamada_kawai(g)
plt.show()
print(g.edges)
print(len(g.edges))