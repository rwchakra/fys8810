# @title Mutual Information Gap Support Functions

import sklearn
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from utils_graphs import pearsonr
#from main import modelArgs, trainArgs, model, test_data
#from main import dataArgs

def compute_mig(z, v):
    if z.shape[0] > 1:

        ## normalize data
        z, z_mean, z_std = normalize_data(z)
        v, v_mean, v_std = normalize_data(v)

        ## discretize data
        z = discretize_data(z)
        v = discretize_data(v)

        m = discrete_mutual_info(z, v)
        assert m.shape[0] == z.shape[0]
        assert m.shape[1] == v.shape[0]
        # m is [num_latents, num_factors]
        entropy = discrete_entropy(v)
        sorted_m = np.sort(m, axis=0)[::-1]

        mig_score = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))

    else:

        mig_score = "MIG not defined for one latent variable"

    return mig_score


## Utilities_______________________________

"""Utility functions that are useful for the different metrics."""


def discrete_mutual_info(z, v):
    """Compute discrete mutual information."""
    num_codes = z.shape[0]
    num_factors = v.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):

            if num_factors > 1:
                m[i, j] = sklearn.metrics.mutual_info_score(v[j, :], z[i, :])
            elif num_factors == 1:
                m[i, j] = sklearn.metrics.mutual_info_score(np.squeeze(v), z[i, :])

    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def discretize_data(target, num_bins=10):
    """Discretization based on histograms."""
    target = np.nan_to_num(target)
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


# @title Correlation between generative parameters v_k and latent variables z_j
def latent_space_encoder(modelArgs, trainArgs, dataArgs, model, data, lowrecon):
    n_samples = 1000
    Attr, A_mod, Param, Topol = data
    Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]

    encoder, decoder = model  ## trained model parts
    z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
    z_var = K.exp(0.5 * z_log_var)

    param_txt = ["n", "p", "attr_param"]
    topol_txt = ["density", "diameter", "assort", "#edges", "avg_degree"]

    if dataArgs["node_attr"] == "none":
        param_txt = param_txt[:2]
        Param = Param[:, :2]

    ## Measuring the Mutual Information Gap ____________________________________________

    v = np.reshape(Param, (Param.shape[1], Param.shape[0]))
    z = np.reshape(z_mean, (z_mean.shape[1], z_mean.shape[0]))

    mig_score = compute_mig(z, v)

    if z_sample.shape[-1] >= 2:

        ## (1) Generative Parameters________________________________________________________

        fig, ax = plt.subplots(nrows=z_sample.shape[-1], ncols=len(param_txt), figsize=(15, 9))
        fig.suptitle('Generative Parameters v' + " – Mutual Information Gap (MIG) score:" + str(round(mig_score, 7)),
                     fontweight="bold")

        for z, row in enumerate(ax):
            for v, col in enumerate(row):
                plt.ylim(-4, 4)
                y = z_sample[:, z]
                x = Param[:, v]
                sns.regplot(x, y, color="steelblue", ax=col, scatter_kws={'alpha': 0.3}, order=2)

                corr = round(pearsonr(x, y)[0], 3)
                cov = round(np.cov(x, y)[0][1] / max(x), 3)
                col.annotate("corr:" + str(corr) + ", cov:" + str(cov), xy=(0, 1), xytext=(12, -12), va='top',
                             xycoords='axes fraction', textcoords='offset points')

        ## add row and column titles
        rows = ['z_{}'.format(row) for row in range(z_sample.shape[-1])]
        cols = [t for t in param_txt]

        for axis, col in zip(ax[0], cols):
            axis.set_title(col, fontweight='bold')

        for axis, row in zip(ax[:, 0], rows):
            axis.set_ylabel(row, rotation=0, size='large', fontweight='bold')

        if lowrecon:
            plt.savefig(
                'mig_'+ '_beta_graphs_' + str(trainArgs['loss_weights'][2]) + '_lowrecon' + '.png')

        else:
            plt.savefig(
                'mig_' + '_beta_graphs_' + str(trainArgs['loss_weights'][2]) + '.png')

        ## (2) Graph Topology_______________________________________________________

        fig, ax = plt.subplots(nrows=z_sample.shape[-1], ncols=len(topol_txt), figsize=(20, 6))
        fig.suptitle('Graph Topology', fontweight="bold")

        for z, row in enumerate(ax):
            for v, col in enumerate(row):
                plt.ylim(-4, 4)
                y = z_sample[:, z]
                x = Topol[:, v]
                sns.regplot(x, y, color="steelblue", ax=col, scatter_kws={'alpha': 0.3})

                corr = round(pearsonr(x, y)[0], 3)
                cov = round(np.cov(x, y)[0][1] / max(x), 3)
                col.annotate("corr:" + str(corr) + ", cov:" + str(cov), xy=(0, 1), xytext=(12, -12), va='top',
                             xycoords='axes fraction', textcoords='offset points')

        ## add row and column titles
        rows = ['z_{}'.format(row) for row in range(z_sample.shape[-1])]
        cols = [t for t in topol_txt]

        for axis, col in zip(ax[0], cols):
            axis.set_title(col, fontweight='bold')

        for axis, row in zip(ax[:, 0], rows):
            axis.set_ylabel(row, rotation=0, size='large', fontweight='bold')

        plt.show()





    elif z_sample.shape[-1] == 1:

        ## (1) Generative Parameters________________________________________________________

        fig, ax = plt.subplots(nrows=z_sample.shape[-1], ncols=len(param_txt), figsize=(20, 5))
        fig.suptitle('Generative Parameters v' + " – Mutual Information Gap (MIG) score: " + str(round(mig_score, 7)),
                     fontweight="bold")

        for v, col in enumerate(range(len(param_txt))):
            plt.sca(ax[v])
            plt.ylim(-4, 4)
            y = z_sample[:, 0]
            x = Param[:, v]
            sns.regplot(x, y, color="steelblue", scatter_kws={'alpha': 0.3}, order=2)

            corr = round(pearsonr(x, y)[0], 3)
            cov = round(np.cov(x, y)[0][1] / max(x), 3)
            plt.annotate("corr:" + str(corr) + ", cov:" + str(cov), xy=(0, 1), xytext=(12, -12), va='top',
                         xycoords='axes fraction', textcoords='offset points')

        ## add row and column titles
        cols = [t for t in param_txt]
        fig.text(0.1, 0.5, 'z_0', fontweight="bold", ha='center', va='center', rotation='vertical')

        for axis, col in zip(ax[:, ], cols):
            axis.set_title(col, fontweight='bold')

        ## (2) Graph Topology_______________________________________________________

        fig, ax = plt.subplots(nrows=z_sample.shape[-1], ncols=len(topol_txt), figsize=(20, 6))

        for v, col in enumerate(range(len(topol_txt))):
            plt.sca(ax[v])
            plt.ylim(-4, 4)
            y = z_sample[:, 0]
            x = Topol[:, v]
            sns.regplot(x, y, color="steelblue", scatter_kws={'alpha': 0.3}, order=2)

            corr = round(pearsonr(x, y)[0], 3)
            cov = round(np.cov(x, y)[0][1] / max(x), 3)
            plt.annotate("corr:" + str(corr) + ", cov:" + str(cov), xy=(0, 1), xytext=(12, -12), va='top',
                         xycoords='axes fraction', textcoords='offset points')

        ## add row and column titles
        cols = [t for t in topol_txt]
        fig.text(0.1, 0.5, 'z_0', fontweight="bold", ha='center', va='center', rotation='vertical')

        for axis, col in zip(ax[:, ], cols):
            axis.set_title(col, fontweight='bold')


## PLOT RESULTS ________________________________________

#latent_space_encoder(modelArgs, trainArgs, dataArgs, model, test_data)