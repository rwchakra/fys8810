# @title Decoder Analysis Support Functions

from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import sys
import networkx as nx
from utils_graphs import unpad_data, plot_graph
#from main import modelArgs, trainArgs
#from main import dataArgs
#from main import model, test_data
import pickle

## DECODER - Latent Space Interpolation____________________________

def generate_manifold(analyzeArgs, modelArgs, trainArgs, model, data, beta_value, lowrecon, maximum_number_of_nodes_n):

    with open('data/data_args' + beta_value, 'rb') as fp:
        dataArgs = pickle.load(fp)

    Attr, A_mod, Param, Topol = data

    encoder, decoder = model  ## trained model parts
    z_mean, z_log_var, z_sample = encoder.predict([Attr, A_mod], trainArgs["batch_size"])
    z_var = K.exp(0.5 * z_log_var)

    if len(analyzeArgs["z"]) >= 2:

        z_sample = np.zeros(modelArgs["latent_dim"])
        z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

        ## fill unobserved dimensions with mean of latent variable dimension
        for dim in range(0, len(z_sample[0])):
            z_sample[0][dim] = np.mean(z_mean[:, dim])

        grid_x = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])
        grid_y = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])[
                 ::-1]  ## revert

        figure = np.zeros((analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"],
                           analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"], 3))
        fig, axs = plt.subplots(analyzeArgs["size_of_manifold"], analyzeArgs["size_of_manifold"], figsize=(8, 8))

        ## Set common labels
        fig.text(0.5, 0.04, "z_" + str(analyzeArgs["z"][0]), ha='center')
        fig.text(0.04, 0.5, "z_" + str(analyzeArgs["z"][1]), va='center', rotation='vertical')

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):

                xi_value = xi ** 1
                yi_value = yi ** 1

                try:
                    z_sample[0][analyzeArgs["z"][0]] = xi ** 1
                    z_sample[0][analyzeArgs["z"][1]] = yi ** 1
                except:
                    print("please select correct latent variables")
                    print("number of latent variables to choose from: z_" + str(np.arange(modelArgs["latent_dim"])))
                    sys.exit()

                #print(z_sample)
                [attr, max_a] = decoder.predict(z_sample)
                # print(attr, max_a)
                g, a, attr = unpad_data(max_a[0], attr[0])
                # print(attr, max_a)
                nodes = g.number_of_nodes()
                edges = g.number_of_edges()
                print((i, j))
                print("p: ", (2 * edges)/(nodes * (nodes - 1)))
                attr = np.clip(attr, 0.0, 1.0)
                fixed_cmap, a_channel = plot_graph(g, max_a[0], attr, draw=False)

                figure[i * dataArgs["max_n_node"]: (i + 1) * dataArgs["max_n_node"],
                j * dataArgs["max_n_node"]: (j + 1) * dataArgs["max_n_node"], :] = a_channel

                plt.sca(axs[i, j])
                nx.draw_kamada_kawai(g, node_size=12, node_color=attr, width=0.2, font_color='white', cmap=fixed_cmap)
                axs[i, j].set_axis_off()

        z_range = abs(analyzeArgs['range'][0])
        if lowrecon:
            plt.savefig('plots/zrange_' + str(z_range) + '_lowrecon_beta_graphs_' + str(trainArgs['loss_weights'][2]) + '.png')

        else:
            #plt.savefig(
                #'plots/zrange_' + str(z_range) + '_beta_graphs_' + str(trainArgs['loss_weights'][2]) + '.png')
            plt.savefig(
                'plots/zrange_' + str(z_range) + '_beta_graphs_' + str(trainArgs['loss_weights'][2])+'_'+str(maximum_number_of_nodes_n) + '.png')
        #plt.show()
        start_range = dataArgs["max_n_node"] // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * dataArgs["max_n_node"] + start_range + 1
        pixel_range = np.arange(start_range, end_range, dataArgs["max_n_node"])
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        print("CREATING FIGURE")
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("z_" + str(analyzeArgs["z"][0]), fontweight='bold')
        plt.ylabel("z_" + str(analyzeArgs["z"][1]), fontweight='bold')



        plt.imshow(figure, cmap='Greys_r')
        if lowrecon:
            plt.savefig('plots/zrange_'+str(z_range)+'_lowrecon_beta_adj_'+str(trainArgs['loss_weights'][2])+'.png')

        else:
            plt.savefig(
                'plots/zrange_' + str(z_range) + '_beta_adj_' + str(trainArgs['loss_weights'][2])+'_'+str(maximum_number_of_nodes_n)+ '.png')

        #plt.show()







    elif len(analyzeArgs["z"]) == 1 or modelArgs["latent_dim"] == 1:

        z_sample = np.zeros(modelArgs["latent_dim"])
        z_sample = np.reshape(z_sample, (1, modelArgs["latent_dim"]))

        ## fill unobserved dimensions with mean of latent variable dimension
        for dim in range(0, len(z_sample[0])):
            z_sample[0][dim] = np.mean(z_mean[:, dim])

        grid_x = np.linspace(analyzeArgs["range"][0], analyzeArgs["range"][1], analyzeArgs["size_of_manifold"])

        figure = np.zeros((1 * dataArgs["max_n_node"], analyzeArgs["size_of_manifold"] * dataArgs["max_n_node"], 3))
        fig, axs = plt.subplots(1, analyzeArgs["size_of_manifold"], figsize=(8, 2))

        ## Set common labels
        fig.text(0.5, 0.04, "z_" + str(analyzeArgs["z"][0]), ha='center')

        axs = axs.ravel()
        for j, xi in enumerate(grid_x):

            xi_value = xi ** 1

            try:
                z_sample[0][analyzeArgs["z"][0]] = xi ** 1
            except:
                print("please select correct latent variables")
                print("number of latent variables to choose from: z_" + str(np.arange(modelArgs["latent_dim"])))
                sys.exit()

            [attr, max_a] = decoder.predict(z_sample)

            g, a, attr = unpad_data(max_a[0], attr[0])
            attr = np.clip(attr, 0.0, 1.0)
            fixed_cmap, a_channel = plot_graph(g, max_a[0], attr, draw=False)

            figure[0:dataArgs["max_n_node"], j * dataArgs["max_n_node"]: (j + 1) * dataArgs["max_n_node"]] = a_channel

            jx = np.unravel_index(j, axs.shape)
            plt.sca(axs[jx])

            nx.draw_kamada_kawai(g, node_size=12, node_color=attr, width=0.2, font_color='white', cmap=fixed_cmap)
            axs[jx].set_axis_off()
            axs[jx].set(ylabel='z_0')

        start_range = dataArgs["max_n_node"] // 2
        end_range = (analyzeArgs["size_of_manifold"] - 1) * dataArgs["max_n_node"] + start_range + 1
        pixel_range = np.arange(start_range, end_range, dataArgs["max_n_node"])
        sample_range_x = np.round(grid_x, 1)

        plt.figure(figsize=(10, 10))
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("z_" + str(analyzeArgs["z"][0]), fontweight='bold')
        plt.imshow(figure, cmap='Greys_r')