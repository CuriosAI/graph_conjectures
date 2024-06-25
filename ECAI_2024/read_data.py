# -*- coding=utf-8 -*-
import numpy as np
import networkx as nx
import random
import time

def vectorize_eig(s,incl_p):
  tokens = s.split()
  read_eig = np.array([float(t) for t in tokens])
  if incl_p:
    return read_eig
    # mantiene l'informazione sulla probabilità con cui è stato generato il grafo
  else:
    return read_eig[:-1]

def built_adj(row):
  row = row.strip() # elimina \n
  G = nx.from_graph6_bytes(row.encode(encoding='UTF-8'))
  A = nx.to_numpy_array(G)
  return A


print("Available datasets for n in [11,30]\n")
print("Select vertices' number: \n")

while True:
  n = input()
  if n < 11 or n > 30:
    print("Unavailable data \n n must be in [11,30]")
  else:
    break

# Lettura grafi e autovalori.
# il dato e il rispettivo target vengono messi in due strutture diverse

g = open("{0}_BrouwerDS.g6".format(n),"r")
e = open("{0}_eigenvalues.txt".format(n),"r")
incl_p = 0

row = g.readline()
G_T = built_adj(row) # tensore dei grafi

# matrice dei target
t = vectorize_eig(e.readline(),incl_p)

eig = 1 # solo per entrare nel loop
row = g.readline()
while row and eig:
  G_T = np.dstack((G_T,built_adj(row)))
  eig = vectorize_eig(e.readline(),incl_p)
  t = np.vstack((t,eig))

  row = g.readline()

g.close()
e.close()