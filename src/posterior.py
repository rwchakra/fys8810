# @title Posterior Distributions

from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy
from utils_graphs import norm, pearsonr
from latent_space_encoder_metrics import compute_mig
import seaborn as sns
import random
from main import modelArgs, trainArgs, test_data, model
from main import dataArgs
from latent_space_decoder import analyzeArgs

def visualize_distributions(modelArgs, trainArgs, model, data):
    n_samples = 1000
    Attr, A_mod, Param, Topol = data
    Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]

    encoder, decoder = model  ## trained model parts
    z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
    z_var = K.exp(0.5 * z_log_var)

    col_titles = ['z_{}'.format(col) for col in range(z_mean.shape[1])]

    fig, ax = plt.subplots(nrows=1, ncols=z_sample.shape[-1], figsize=(10, 3))

    if z_sample.shape[-1] > 1:
        for z, col in enumerate(ax):
            plt.sca(ax[z])
            col.yaxis.set_visible(False)
            plt.xlabel('z_' + str(z), fontweight="bold")
            grid = np.linspace(-4, 4, 1000)
            kde_z = scipy.stats.gaussian_kde(z_sample[:, z])

            plt.plot(grid, norm.pdf(grid, 0.0, 1.0), label="Gaussian prior", color='steelblue', linestyle=':',
                     markerfacecolor='blue', linewidth=6)
            plt.plot(grid, kde_z(grid), label="z", color='midnightblue', markerfacecolor='blue', linewidth=6)

    else:
        plt.yticks([])
        plt.xlabel('z_0', fontweight="bold")
        grid = np.linspace(-4, 4, 1000)
        kde_z = scipy.stats.gaussian_kde(z_sample[:, 0])

        plt.plot(grid, norm.pdf(grid, 0.0, 1.0), label="Gaussian prior", color='steelblue', linestyle=':',
                 markerfacecolor='blue', linewidth=6)
        plt.plot(grid, kde_z(grid), label="z", color='midnightblue', markerfacecolor='blue', linewidth=6)


def attr_topol_correlation(analyzeArgs, modelArgs, trainArgs, dataArgs, model, data):
    n_samples = 1000
    Attr, A_mod, Param, Topol = data
    Attr, A_mod, Param, Topol = Attr[:n_samples], A_mod[:n_samples], Param[:n_samples], Topol[:n_samples]

    param_txt = ["n", "p", "attr_param"]

    ## Randomize Attributes ___________________________________________

    Attr_rand = np.copy(Attr)
    Rand_degree = np.zeros((Attr_rand.shape[0]))

    for i in range(Attr_rand.shape[0]):

        rand_degree = random.uniform(0.0, 1.0)
        attr_rand = Attr_rand[i]

        ## reshape attr and unpad
        attr_rand = np.reshape(attr_rand, (attr_rand.shape[0]))
        nodes_n = attr_rand[attr_rand > 0.0].shape[0]
        attr_rand = attr_rand[:nodes_n]  ## shorten

        if analyzeArgs["randomization"] == "shuffle attributes":
            # math.ceil
            for m in range(0, int(rand_degree * nodes_n)):
                swap = np.random.randint(low=0, high=attr_rand.shape[0], size=2)
                temp = attr_rand[swap[0]]
                attr_rand[swap[0]] = attr_rand[swap[1]]
                attr_rand[swap[1]] = temp

        elif analyzeArgs["randomization"] == "assign random attributes":

            rand_n = np.random.choice(nodes_n, int(rand_degree * nodes_n), replace=False)
            rand_value = np.random.uniform(0, 1, rand_n.shape[0])

            attr_rand[rand_n] = rand_value

        ## pad features with zeroes

        zeroes = np.zeros((dataArgs["max_n_node"] - attr_rand.shape[0]))
        attr_rand = np.concatenate((attr_rand, zeroes))
        attr_rand = np.reshape(attr_rand, (attr_rand.shape[-1], 1))

        Attr_rand[i] = attr_rand
        Rand_degree[i] = rand_degree

    ## Encoder ______________________________________________

    encoder, decoder = model  ## trained model parts

    ## 1) Original Attributes
    z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
    z_var = K.exp(0.5 * z_log_var)

    ## 2) Randomized Attributes
    z_mean_rand, z_log_var_rand, z_sample_rand = encoder.predict([Attr_rand, A_mod], trainArgs["batch_size"])
    z_var_rand = K.exp(0.5 * z_log_var_rand)

    z_shift = np.abs(z_sample - z_sample_rand)
    v_rand = Rand_degree

    ## Measuring the Mutual Information Gap ____________________________________________

    v_rand_reshaped = np.reshape(v_rand, (1, v_rand.shape[0]))
    z_shift_reshaped = np.reshape(z_shift, (z_shift.shape[1], z_shift.shape[0]))

    mig_score = compute_mig(z_shift_reshaped, v_rand_reshaped)
    # print("Mutual Information Gap (MIG) score:", round(mig_score, 7))

    # sys.exit()

    ## Latent Variables and Attribute Shift ____________________________

    fig, ax = plt.subplots(nrows=1, ncols=z_shift.shape[-1], figsize=(15, 6))
    fig.suptitle(
        'Correlation between Node Attributes and Latent Variables' + " – Mutual Information Gap (MIG) score: " + str(
            round(mig_score, 7)), fontweight="bold")

    for latent_z, col in enumerate(ax):
        plt.sca(ax[latent_z])
        plt.ylim(0, 3)
        y = z_shift[:, latent_z]
        x = v_rand
        sns.regplot(x, y, color="steelblue", ax=col, ci=None, x_ci='sd', scatter_kws={'alpha': 0.3}, order=2)

        ## compute correlation and standardized covariance
        corr = round(pearsonr(x, y)[0], 3)
        cov = round(np.cov(x, y)[0][1] / max(x), 3)
        std = round(np.std(y), 3)
        col.annotate("corr:" + str(corr) + ", std:" + str(std), xy=(0, 1), xytext=(12, -12), va='top',
                     xycoords='axes fraction', textcoords='offset points', fontweight='bold')
        col.set_xlabel('attribute randomization degree', fontweight="bold")
        col.set_ylabel('shift in z_' + str(latent_z), rotation=90, fontweight="bold")


if dataArgs["node_attr"] == "none":
    print("generated graphs do not feature node attributes")
else:
    attr_topol_correlation(analyzeArgs, modelArgs, trainArgs, dataArgs, model, test_data)

visualize_distributions(modelArgs, trainArgs, model, test_data)

# @title Dependency between Node Attributes and Graph Topology


analyzeArgs = dict()

# @markdown select type of node attribute randomization (note: shuffling will not have any effects if all nodes attributes in a graph are the same)

attribute_randomization = "assign random attributes"  # @param ["shuffle attributes", "assign random attributes"]
analyzeArgs["randomization"] = attribute_randomization