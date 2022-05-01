# @title Graph Generation Methods
import networkx as nx
from networkx.generators import random_graphs
import numpy as np
from tqdm import tqdm
import random
from utils_graphs import sort_adjacency, pad_data, unpad_data, plot_graph

def generate_graph(n, p):
    g = random_graphs.erdos_renyi_graph(n, p, seed=None, directed=False)
    a = nx.adjacency_matrix(g)

    return g, a


def compute_topol(g):
    density = nx.density(g)

    if nx.is_connected(g):
        diameter = nx.diameter(g)
    else:
        diameter = -1

    cluster_coef = nx.average_clustering(g)

    # if g.number_of_edges() > 2 and len(g) > 2:
    #    assort = nx.degree_assortativity_coefficient(g, x='out', y='in')
    # else:
    #    assort = 0

    edges = g.number_of_edges()
    avg_degree = sum(i for i in nx.degree_centrality(g).values()) / len(nx.degree_centrality(g).keys())

    topol = [density, diameter, cluster_coef, edges, avg_degree]

    return topol


def generate_attr(g, n, p, dataArgs):
    if dataArgs["node_attr"] == "none":
        attr = np.ones((n)) * 0.5

        attr_param = 0

    if dataArgs["node_attr"] == "random":
        attr = np.random.rand(n)

        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "degree":
        attr = np.asarray([int(x[1]) for x in sorted(g.degree())])
        attr = (attr + 1) / (dataArgs["max_n_node"] + 1)

        attr_param = np.random.rand(1)

    if dataArgs["node_attr"] == "uniform":
        uniform_attr = np.random.rand(1)
        attr = np.ones((n)) * uniform_attr

        attr_param = uniform_attr

    if dataArgs["node_attr"] == "p_value":
        attr = np.ones((n)) * p

        attr_param = p

    return g, attr, attr_param


def generate_data(dataArgs):
    A = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], dataArgs["max_n_node"], 1))  ## graph data
    Attr = np.zeros((dataArgs["n_graph"], dataArgs["max_n_node"], 1))  ## graph data
    Param = np.zeros((dataArgs["n_graph"], 3))  ## generative parameters
    Topol = np.zeros((dataArgs["n_graph"], 5))  ## topological properties

    for i in tqdm(range(0, dataArgs["n_graph"])):
        n = random.randint(1, dataArgs["max_n_node"])  ## generate number of nodes n between 1 and max_n_node and
        p = random.uniform(dataArgs["p_range"][0], dataArgs["p_range"][1])  ## floating p from range

        g, a = generate_graph(n, p)
        g, attr, attr_param = generate_attr(g, n, p, dataArgs)

        g, a, attr = sort_adjacency(g, a, attr)  ## extended BOSAM sorting algorithm
        a, attr = pad_data(a, attr, dataArgs[
            "max_n_node"])  ## pad adjacency matrix to allow less nodes than max_n_node and fill diagonal

        A[i] = a
        Attr[i] = attr
        Param[i] = [n, p, attr_param]
        Topol[i] = compute_topol(g)

    return A, Attr, Param, Topol


