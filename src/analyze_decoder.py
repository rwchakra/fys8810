#@title  Interpolation Manifold Specifications
from tensorflow import keras
import numpy as np
import pickle
from keras import backend as K
from models import sampling
from latent_space_decoder import generate_manifold

beta_value = "5"
lowrecon = False
maximum_number_of_nodes_n = 24

if lowrecon:
  with open('allArgs/args_lowrecon_beta_'+beta_value, 'rb') as f:
    data = pickle.load(f)
    trainArgs, modelArgs = data

else:
  with open('allArgs/args_beta_'+beta_value+'_'+str(maximum_number_of_nodes_n), 'rb') as f:
    data = pickle.load(f)
    trainArgs, modelArgs = data

if lowrecon:

  with open('data/train_test_lowrecon', 'rb') as f:
    data = pickle.load(f)
    A_train = data[0]
    A_test = data[4]
    A_train_mod = data[8]
    A_test_mod = data[9]
    Attr_train = data[1]
    Attr_test = data[5]
    test_data = data[11]

else:
  with open('data/train_test'+'_'+str(maximum_number_of_nodes_n), 'rb') as f:
    data = pickle.load(f)
    A_train = data[0]
    A_test = data[4]
    A_train_mod = data[8]
    A_test_mod = data[9]
    Attr_train = data[1]
    Attr_test = data[5]
    test_data = data[11]

#vae=VAE(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test)

if lowrecon:

  model = keras.models.load_model('saved_models/encoder_lowrecon_beta_'+beta_value, custom_objects={'node_invariant_x':lambda x: K.mean(x, axis=1), 'sampling': sampling}), keras.models.load_model('saved_models/decoder_lowrecon_beta_'+beta_value)

else:
  model = keras.models.load_model('saved_models/encoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n),
                                  custom_objects={'node_invariant_x': lambda x: K.mean(x, axis=1),
                                                  'sampling': sampling}), keras.models.load_model(
    'saved_models/decoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))

analyzeArgs = {"z": [0, 2], "act_range": [-6, 6], "act_scale": 1, "size_of_manifold": 7}

#@markdown select one or two latent variables to visualize (limited by number of variables used in model)

z_vars = list()
z_0 = False #@param {type:"boolean"}
if z_0:
  z_vars.append(0)
z_1 = False #@param {type:"boolean"}
if z_1:
  z_vars.append(1)
z_2 = True #@param {type:"boolean"}
if z_2:
  z_vars.append(2)
z_3 = True #@param {type:"boolean"}
if z_3:
  z_vars.append(3)
z_4 = False #@param {type:"boolean"}
if z_4:
  z_vars.append(4)
z_5 = False #@param {type:"boolean"}
if z_5:
  z_vars.append(5)

#analyzeArgs["z"] = np.asarray(z_vars)

print("the trained model comprises " + str(modelArgs["latent_dim"]) + " latent variables from which z_" + str(analyzeArgs["z"]) + " are visualized.\n\n")


#@markdown manifold settings
interpolation_range =  "-2,2" #@param [[-2,2], [-4, 4], [-6, 6]]
size_of_manifold = "5" #@param [5, 7, 10, 15]

analyzeArgs["range"] = [int(interpolation_range.split(",")[0]),int(interpolation_range.split(",")[1])]
analyzeArgs["size_of_manifold"] = int(size_of_manifold)

generate_manifold(analyzeArgs, modelArgs, trainArgs, model, test_data, beta_value, lowrecon, maximum_number_of_nodes_n)