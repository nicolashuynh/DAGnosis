# third party
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(A: np.ndarray, base_node: int):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    color_map = ["grey" for i in range(len(A))]
    color_map[base_node] = "red"
    nx.draw_kamada_kawai(G, node_color=color_map, with_labels=True)
    plt.show()
