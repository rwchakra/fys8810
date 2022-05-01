from latent_space_encoder_metrics import *
from tensorflow import keras
from keras import backend as K
from models import sampling
import pickle

beta_value = "10"
lowrecon = False
maximum_number_of_nodes_n = 24

if lowrecon:
  with open('allArgs/args_lowrecon_beta_'+beta_value+'_'+str(maximum_number_of_nodes_n), 'rb') as f:
    data = pickle.load(f)
    trainArgs, modelArgs = data

else:
  with open('allArgs/args_beta_'+beta_value+'_'+str(maximum_number_of_nodes_n), 'rb') as f:
    data = pickle.load(f)
    trainArgs, modelArgs = data

if lowrecon:

  with open('data/train_test_lowrecon'+'_'+str(maximum_number_of_nodes_n), 'rb') as f:
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

  model = keras.models.load_model('saved_models/encoder_lowrecon_beta_'+beta_value+'_'+str(maximum_number_of_nodes_n), custom_objects={'node_invariant_x':lambda x: K.mean(x, axis=1), 'sampling': sampling}), keras.models.load_model('saved_models/decoder_lowrecon_beta_'+beta_value+'_'+str(maximum_number_of_nodes_n))

else:
  model = keras.models.load_model('saved_models/encoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n),
                                  custom_objects={'node_invariant_x': lambda x: K.mean(x, axis=1),
                                                  'sampling': sampling}), keras.models.load_model(
    'saved_models/decoder_beta_' + beta_value+'_'+str(maximum_number_of_nodes_n))

#analyzeArgs = {"z": [0, 2], "act_range": [-6, 6], "act_scale": 1, "size_of_manifold": 7}
dataArgs = {'node_attr': 'uniform'}
latent_space_encoder(modelArgs, trainArgs, dataArgs, model, test_data)