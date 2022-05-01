#@title  Model Specifications
import keras.models

from models import VAE, build_model
from keras import backend as K
import numpy as np
from utils_models import preprocess_adj_tensor_with_identity
from generate_graphs import generate_data
from utils_graphs import unpad_data, plot_graph
import pickle
from latent_space_encoder_metrics import latent_space_encoder
from latent_space_decoder import generate_manifold
#FOR FALSE FRESH START, CHANGE VALUE OF BETA ONLY AND RUN MAIN.PY
import time

#@markdown model build specifications

print("INSIDE MAIN")

fresh_start = True
lowrecon = False

modelArgs = {"gnn_filters": 2, "conv_filters": 16, "kernel_size": 3}

number_of_latent_variables= "4" #@param [1, 2, 3, 4, 5]
modelArgs["latent_dim"] = int(number_of_latent_variables)

# @title  ER Random Graph Data Specifications

dataArgs = dict()

# @markdown select the maximum number of nodes per graph

maximum_number_of_nodes_n = "24"  # @param [12, 24, 30, 48]
dataArgs["max_n_node"] = int(maximum_number_of_nodes_n)

# @markdown select the range of p

range_of_linkage_probability_p = "0,1"  # @param [[0.0,1.0], [0.2,0.8], [0.5,0.5]]
dataArgs["p_range"] = [float(range_of_linkage_probability_p.split(",")[0]),
                       float(range_of_linkage_probability_p.split(",")[1])]

# @markdown specify the generation process of node attributes

node_attributes = "uniform"  # @param ["none", "uniform", "degree", "p_value", "random"]
dataArgs["node_attr"] = node_attributes

# @markdown specify the number of graphs generated for training and validation
number_of_graph_instances = "50000"  # @param [1, 100, 1000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000]
dataArgs["n_graph"] = int(number_of_graph_instances)



#g, a, attr = unpad_data(A[0], Attr[0])
#fixed_cmap, a_channel = plot_graph(g, A[0], attr, draw=True)

#@markdown training specifications

trainArgs = dict()

weight_graph_reconstruction_loss = "5" #@param [0, 1, 2, 3, 5, 10, 20]
weight_attribute_reconstruction_loss = "2" #@param [0, 1, 2, 3, 5, 10, 20]
beta_value = "5" #@param [0, 1, 2, 3, 5, 10, 20]
trainArgs["loss_weights"] = [float(weight_graph_reconstruction_loss), float(weight_attribute_reconstruction_loss), int(beta_value)]

epochs = "500" #@param [10, 20, 50]
trainArgs["epochs"] = int(epochs)
batch_size = "1024" #@param [2, 4, 8, 16, 32, 128, 512, 1024]
trainArgs["batch_size"] = int(batch_size)
early_stop = "2" #@param [1, 2, 3, 4, 10]
trainArgs["early_stop"] = int(early_stop)
train_test_split = "0.1" #@param [0.1, 0.2, 0.3, 0.5]
trainArgs["data_split"] = float(train_test_split)

if lowrecon:
    with open('data/data_lowrecon_args' + beta_value+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
        pickle.dump(dataArgs, fp)

else:
    with open('data/data_args' + beta_value, 'wb') as fp:
        pickle.dump(dataArgs, fp)

## Train and Test Split (IF FIRST RUN, ELSE LOAD)_______________________________________________

print("RUNNING FOR BETA: ", beta_value)
print("LAMBDA 1: ", weight_graph_reconstruction_loss)
print("SCALING: ", not lowrecon)
if fresh_start:

    A, Attr, Param, Topol = generate_data(dataArgs)
    A_train = A[:int((1-trainArgs["data_split"])*A.shape[0])]
    Attr_train = Attr[:int((1-trainArgs["data_split"])*Attr.shape[0])]
    Param_train = Param[:int((1-trainArgs["data_split"])*Param.shape[0])]
    Topol_train = Topol[:int((1-trainArgs["data_split"])*Topol.shape[0])]

    A_test = A[int((1-trainArgs["data_split"])*A.shape[0]):]
    Attr_test = Attr[int((1-trainArgs["data_split"])*Attr.shape[0]):]
    Param_test = Param[int((1-trainArgs["data_split"])*Param.shape[0]):]
    Topol_test = Topol[int((1-trainArgs["data_split"])*Topol.shape[0]):]

    ## build graph_conv_filters

    SYM_NORM = True
    A_train_mod = preprocess_adj_tensor_with_identity(np.squeeze(A_train), SYM_NORM)
    A_test_mod = preprocess_adj_tensor_with_identity(np.squeeze(A_test), SYM_NORM)

    train_data = (Attr_train, A_train_mod, Param_train, Topol_train)
    test_data = (Attr_test, A_test_mod, Param_test, Topol_test)

    #save for all of the above
    data = [A_train, Attr_train, Param_train, Topol_train, A_test, Attr_test, Param_test, Topol_test,
            A_train_mod, A_test_mod, train_data, test_data]

    if lowrecon:
        with open('./data/train_test_lowrecon'+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(data, fp)

    else:
        with open('./data/train_test_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(data, fp)


    modelArgs["input_shape"], modelArgs["output_shape"] = ((Attr_train.shape[1], 1),
                                                           (A_train.shape[1], A_train.shape[2], 1)), (
                                                              (Attr_test.shape[1], 1),
                                                              (A_test.shape[1], A_test.shape[2], 1))

    start = time.time()
    vae = VAE(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)
    # ## TRAIN______________________________________________
    #
    # # Set callback functions to early stop training and save the best model so far
    # # callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    #
    # vae = build_model(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)
    # vae.compile(optimizer='adam', loss=loss_func)
    # vae.fit([Attr_train, A_train_mod], [Attr_train, A_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"],
    # validation_data=([Attr_test, A_test_mod], [Attr_test, A_test]))
    encoder, decoder = vae.model

    if lowrecon:
        encoder.save('saved_models/encoder_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))
        decoder.save('saved_models/decoder_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))

    else:
        encoder.save('saved_models/encoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))
        decoder.save('saved_models/decoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))



    all_args = [trainArgs, modelArgs]

    # Save trainArgs and modelArgs

    if lowrecon:
        with open('allArgs/args_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(all_args, fp)

    else:
        with open('allArgs/args_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(all_args, fp)



else:
    #load for all of them

    if lowrecon:
        with open('data/train_test_lowrecon'+'_'+str(maximum_number_of_nodes_n), 'rb') as fp:
            data = pickle.load(fp)

    else:
        with open('data/train_test'+'_'+str(maximum_number_of_nodes_n), 'rb') as fp:
            data = pickle.load(fp)

    A_train = data[0]
    Attr_train = data[1]
    Param_train = data[2]
    Topol_train = data[3]
    A_test = data[4]
    Attr_test = data[5]
    Param_test = data[6]
    Topol_test = data[7]
    A_train_mod = data[8]
    A_test_mod = data[9]
    train_data = data[10]
    test_data = data[11]

    modelArgs["input_shape"], modelArgs["output_shape"] = ((Attr_train.shape[1], 1),
                                                           (A_train.shape[1], A_train.shape[2], 1)), (
                                                              (Attr_test.shape[1], 1),
                                                              (A_test.shape[1], A_test.shape[2], 1))

    vae = VAE(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)
    # ## TRAIN______________________________________________
    #
    # # Set callback functions to early stop training and save the best model so far
    # # callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    #
    # vae = build_model(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)
    # vae.compile(optimizer='adam', loss=loss_func)
    # vae.fit([Attr_train, A_train_mod], [Attr_train, A_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"],
    # validation_data=([Attr_test, A_test_mod], [Attr_test, A_test]))
    encoder, decoder = vae.model

    if lowrecon:
        encoder.save('saved_models/encoder_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))
        decoder.save('saved_models/decoder_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))

    else:
        encoder.save('saved_models/encoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))
        decoder.save('saved_models/decoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))


    all_args = [trainArgs, modelArgs]

    # Save trainArgs and modelArgs

    if lowrecon:
        with open('allArgs/args_lowrecon_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(all_args, fp)

    else:
        with open('allArgs/args_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n), 'wb') as fp:
            pickle.dump(all_args, fp)


#analyzeArgs = {"z": [0, 1], "act_range": [-6, 6], "act_scale": 1, "size_of_manifold": 7}

latent_space_encoder(modelArgs, trainArgs, dataArgs, vae.model, test_data, lowrecon)
end = time.time()
print("Time taken: ", (end - start))
#generate_manifold(analyzeArgs, modelArgs, trainArgs, vae.model, data, beta_value, lowrecon, maximum_number_of_nodes_n)

