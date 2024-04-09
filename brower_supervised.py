from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import pandas as pd
import networkx as nx

ROOT_FOLDER = "D:\\Dataset grafi\BrouwerData\\"
TOT_MINIBATCH = 101
FIRST_MINIBATCH=range(0,1)
TRAIN_MINIBATCHES = range(0,70)
VALIDATION_MINIBATCHES = range(70,85)
TEST_MINIBATCHES = range(85,101)
NODES_NUM = 11

def load_dataset(indexes):
    data_list = []
    for id in indexes:
        graph_file = f"{ROOT_FOLDER}{id}_minibatch.g6"
        target_file = f"{ROOT_FOLDER}{id}_minibatch_targets.txt"
        Gs = nx.read_graph6(graph_file)
        ssv = pd.read_csv(target_file, delim_whitespace=" ",header=None)
        for (G, row) in zip(Gs, ssv.iloc):
            edge_list = nx.to_edgelist(G)
            data = Data(edge_index=edge_list)
            data.num_nodes = NODES_NUM
            data.eigenvalues = row[0:NODES_NUM]
            data.probability = row[NODES_NUM]
            data_list.append(data)
    return data_list



# prepare networks
mlp = nn.Sequential(nn.Linear((NODES_NUM,1),(NODES_NUM,NODES_NUM)), nn.ReLU(), nn.Linear((NODES_NUM, NODES_NUM),(NODES_NUM,1)))


# batch_size = 32
# first_list = load_dataset(FIRST_MINIBATCH)
# first_loader = DataLoader(first_list, batch_size=batch_size, shuffle=True)
# for batch in first_loader:
#     print(batch)
# # train_list = load_dataset(TRAIN_MINIBATCHES)
# # validation_list = load_dataset(VALIDATION_MINIBATCHES)
# # test_list = load_dataset(TEST_MINIBATCHES)
