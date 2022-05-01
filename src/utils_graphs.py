# @title Libraries and Support Functions
## Basic
from tqdm import tqdm
import argparse
import os
import random
import itertools
import sys
import math

## Computation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.stats.stats import pearsonr
from scipy.stats import norm

## Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

## Network Processing
import networkx as nx
from networkx.generators import random_graphs


def sort_adjacency(g, a, attr):
    node_k1 = dict(g.degree())  ## sort by degree
    node_k2 = nx.average_neighbor_degree(g)  ## sort by neighbor degree
    node_closeness = nx.closeness_centrality(g)
    node_betweenness = nx.betweenness_centrality(g)

    node_sorting = list()

    for node_id in g.nodes():
        node_sorting.append(
            (node_id, node_k1[node_id], node_k2[node_id], node_closeness[node_id], node_betweenness[node_id]))

    node_descending = sorted(node_sorting, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    mapping = dict()

    for i, node in enumerate(node_descending):
        mapping[node[0]] = i

        temp = attr[node[0]]  ## switch node attributes according to sorting
        attr[node[0]] = attr[i]
        attr[i] = temp

    a = nx.adjacency_matrix(g, nodelist=mapping.keys()).todense()  ## switch graph node ids according to sorting

    return g, a, attr


def pad_data(a, attr, max_n_node):
    np.fill_diagonal(a, 1.0)  ## fill the diagonal with fill_diag

    max_a = np.zeros([max_n_node, max_n_node])
    max_a[:a.shape[0], :a.shape[1]] = a
    max_a = np.expand_dims(max_a, axis=2)

    zeroes = np.zeros((max_n_node - attr.shape[0]))
    attr = np.concatenate((attr, zeroes))

    attr = np.expand_dims(attr, axis=1)

    return max_a, attr


def unpad_data(max_a, attr):
    keep = list()
    max_a = np.reshape(max_a, (max_a.shape[0], max_a.shape[1]))

    max_a[max_a > 0.5] = 1.0
    max_a[max_a <= 0.5] = 0.0

    for i in range(0, max_a.shape[0]):
        if max_a[i][i] > 0:
            keep.append(i)
    # print(len(keep))
    ## delete rows and columns
    a = max_a
    a = a[:, keep]  # keep columns
    a = a[keep, :]  # keep rows

    attr = np.reshape(attr, (attr.shape[0]))

    attr = attr[:len(keep)]  ## shorten
    g = nx.from_numpy_matrix(a)

    return g, a, attr


def plot_graph(g, a, attr, draw):
    orig_cmap = plt.cm.PuBu
    fixed_cmap = shiftedColorMap(orig_cmap, start=min(attr), midpoint=0.5, stop=max(attr), name='fixed')

    ## adjust colour reconstructed_a_padded according to features
    a = np.reshape(a, (a.shape[0], a.shape[1]))
    a_channel = np.copy(a)
    a_channel = np.tile(a_channel[:, :, None], [1, 1, 3])  ## broadcast 1 channel to 3

    for node in range(0, len(g)):
        color = fixed_cmap(attr[node])[:3]
        a_channel[node, :node + 1] = a_channel[node, :node + 1] * color
        a_channel[:node, node] = a_channel[:node, node] * color

    if draw == True:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        # plt.axis('off')
        plt.sca(axes[0])
        nx.draw_kamada_kawai(g, node_color=attr, font_color='white', cmap=fixed_cmap)
        axes[1].set_axis_off()
        axes[1].imshow(a_channel)
        fig.tight_layout()

    return fixed_cmap, a_channel


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

