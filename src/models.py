# @title Model Support Functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## Keras
import tensorflow as tf

from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout, Activation, \
    concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.layers import Layer, InputSpec

from utils_models import graph_conv_op, preprocess_adj_tensor_with_identity
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import numpy as np


class MultiGraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiGraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        if self.num_filters != int(input_shape[1][-2]/input_shape[1][-1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

        self.input_dim = input_shape[0][-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):

        output = graph_conv_op(inputs[0], self.num_filters, inputs[1], self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MultiGraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# @title Build and Train Model
class VAE():

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps

    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def __init__(self, modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test):

        ## MODEL ______________________________________________________________

        ## Graph Neural Network Architecture __________________________________

        ## 1) build encoder model____________________________________

        # build graph_conv_filters
        SYM_NORM = True
        num_filters = modelArgs['gnn_filters']
        graph_conv_filters = preprocess_adj_tensor_with_identity(np.squeeze(A_train), SYM_NORM)

        # build model
        X_input = Input(shape=(Attr_train.shape[1], Attr_train.shape[2]), name="node_attributes")
        graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]),
                                         name="adjacency_matrix")

        # define inputs of features and graph topologies
        inputs = [X_input, graph_conv_filters_input]

        x = MultiGraphCNN(100, num_filters, activation='elu')([X_input, graph_conv_filters_input])
        x = Dropout(0.1)(x)
        x = MultiGraphCNN(100, num_filters, activation='elu')([x, graph_conv_filters_input])
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.mean(x, axis=1))(
            x)  # adding a node invariant layer to make sure output does not depend upon the node order in a graph.
        x = Dense(8, activation='relu')(x)
        x = Dense(6, activation='relu')(x)

        z_mean = Dense(modelArgs["latent_dim"], name='z_mean')(x)
        z_log_var = Dense(modelArgs["latent_dim"], name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(modelArgs["latent_dim"],), name='z')([z_mean, z_log_var])

        latent_inputs = Input(shape=(modelArgs["latent_dim"],), name='z_sampling')

        ## 2.1) build attribute decoder model __________________________

        y = Dense(4, activation='relu')(latent_inputs)
        y = Dense(6, activation='relu')(latent_inputs)
        y = Dense(10, activation='relu')(latent_inputs)
        y = Dense(modelArgs["output_shape"][0][0], activation='sigmoid')(y)
        attr_output = Reshape(modelArgs["output_shape"][0], name='node_attributes')(y)

        ## 2.2) build adjacency decoder model __________________________

        ## shape info needed to build decoder model
        x_2D = Input(shape=modelArgs["input_shape"][1], name='adjacency_decoder')

        for i in range(2):
            modelArgs['conv_filters'] *= 2
            x_2D = Conv2D(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'], activation='relu',
                          strides=2, padding='same')(x_2D)
        shape_2D = K.int_shape(x_2D)

        x_2D = Dense(shape_2D[1] * shape_2D[2] * shape_2D[3], activation='relu')(latent_inputs)
        x_2D = Reshape((shape_2D[1], shape_2D[2], shape_2D[3]))(x_2D)

        for i in range(2):
            x_2D = Conv2DTranspose(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'],
                                   activation='relu', strides=2, padding='same')(x_2D)
            modelArgs['conv_filters'] //= 2

        a_output = Conv2DTranspose(filters=1, kernel_size=modelArgs['kernel_size'], activation='sigmoid',
                                   padding='same', name='adjacency_matrix')(x_2D)

        ## INSTANTIATE___________________________________

        ## 1) instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # encoder.summary()

        ## 2) instantiate decoder model
        decoder = Model(latent_inputs, [attr_output, a_output], name='reconstruction')
        # decoder.summary()

        ## 3) instantiate VAE model
        attr_a_outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, attr_a_outputs, name='vae')

        ## LOSS FUNCTIONS ______________________________________

        def loss_func(y_true, y_pred):

            y_true_attr = y_true[0]
            y_pred_attr = y_pred[0]

            y_true_a = y_true[1]
            y_pred_a = y_pred[1]

            ## ATTR RECONSTRUCTION LOSS_______________________
            ## mean squared error
            attr_reconstruction_loss = mse(K.flatten(y_true_attr), K.flatten(y_pred_attr))
            # Scaling below by input shape (Why?)
            attr_reconstruction_loss *= modelArgs["input_shape"][0][0]

            ## A RECONSTRUCTION LOSS_______________________
            ## binary cross-entropy
            a_reconstruction_loss = binary_crossentropy(K.flatten(y_true_a), K.flatten(y_pred_a))
            # Scaling below by input shape (Why?)
            a_reconstruction_loss *= (modelArgs["input_shape"][1][0] * modelArgs["input_shape"][1][1])

            ## KL LOSS _____________________________________________
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            ## COMPLETE LOSS __________________________________________________

            loss = K.mean(trainArgs["loss_weights"][0] * a_reconstruction_loss + trainArgs["loss_weights"][
                1] * attr_reconstruction_loss + trainArgs["loss_weights"][2] * kl_loss)

            return loss

        ## MODEL COMPILE______________________________________________

        vae.compile(optimizer='adam', loss=loss_func)
        # vae.summary()

        # ## TRAIN______________________________________________
        #
        # # Set callback functions to early stop training and save the best model so far
        # # callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        #
        vae.fit([Attr_train, A_train_mod], [Attr_train, A_train], epochs=trainArgs["epochs"], batch_size=trainArgs["batch_size"],
                 validation_data=([Attr_test, A_test_mod], [Attr_test, A_test]))


        # beta_value = str(trainArgs['loss_weights'][2])
        #
        # encoder.save('saved_models/encoder_beta_' + beta_value)
        # decoder.save('saved_models/decoder_beta_' + beta_value)
        #
        # model = keras.models.load_model('saved_models/encoder_beta_' + beta_value + '/saved_model.pb',
        #                                 custom_objects={'loss_func': loss_func}), keras.models.load_model(
        #     'saved_models/decoder_beta_' + beta_value + '/saved_model.pb'

        self.model = (encoder, decoder)

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_model(modelArgs, trainArgs, A_train, A_test, A_train_mod, A_test_mod, Attr_train, Attr_test):

    ## MODEL ______________________________________________________________

    ## Graph Neural Network Architecture __________________________________

    ## 1) build encoder model____________________________________

    # build graph_conv_filters
    SYM_NORM = True
    num_filters = modelArgs['gnn_filters']
    graph_conv_filters = preprocess_adj_tensor_with_identity(np.squeeze(A_train), SYM_NORM)

    # build model
    X_input = Input(shape=(Attr_train.shape[1], Attr_train.shape[2]), name="node_attributes")
    graph_conv_filters_input = Input(shape=(graph_conv_filters.shape[1], graph_conv_filters.shape[2]),
                                     name="adjacency_matrix")

    # define inputs of features and graph topologies
    inputs = [X_input, graph_conv_filters_input]

    x = MultiGraphCNN(100, num_filters, activation='elu')([X_input, graph_conv_filters_input])
    x = Dropout(0.1)(x)
    x = MultiGraphCNN(100, num_filters, activation='elu')([x, graph_conv_filters_input])
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(
        x)  # adding a node invariant layer to make sure output does not depend upon the node order in a graph.
    x = Dense(8, activation='relu')(x)
    x = Dense(6, activation='relu')(x)

    z_mean = Dense(modelArgs["latent_dim"], name='z_mean')(x)
    z_log_var = Dense(modelArgs["latent_dim"], name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(modelArgs["latent_dim"],), name='z')([z_mean, z_log_var])

    latent_inputs = Input(shape=(modelArgs["latent_dim"],), name='z_sampling')

    ## 2.1) build attribute decoder model __________________________

    y = Dense(4, activation='relu')(latent_inputs)
    y = Dense(6, activation='relu')(latent_inputs)
    y = Dense(10, activation='relu')(latent_inputs)
    y = Dense(modelArgs["output_shape"][0][0], activation='sigmoid')(y)
    attr_output = Reshape(modelArgs["output_shape"][0], name='node_attributes')(y)

    ## 2.2) build adjacency decoder model __________________________

    ## shape info needed to build decoder model
    x_2D = Input(shape=modelArgs["input_shape"][1], name='adjacency_decoder')

    for i in range(2):
        modelArgs['conv_filters'] *= 2
        x_2D = Conv2D(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'], activation='relu',
                      strides=2, padding='same')(x_2D)
    shape_2D = K.int_shape(x_2D)

    x_2D = Dense(shape_2D[1] * shape_2D[2] * shape_2D[3], activation='relu')(latent_inputs)
    x_2D = Reshape((shape_2D[1], shape_2D[2], shape_2D[3]))(x_2D)

    for i in range(2):
        x_2D = Conv2DTranspose(filters=modelArgs['conv_filters'], kernel_size=modelArgs['kernel_size'],
                               activation='relu', strides=2, padding='same')(x_2D)
        modelArgs['conv_filters'] //= 2

    a_output = Conv2DTranspose(filters=1, kernel_size=modelArgs['kernel_size'], activation='sigmoid',
                               padding='same', name='adjacency_matrix')(x_2D)

    ## INSTANTIATE___________________________________

    ## 1) instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()

    ## 2) instantiate decoder model
    decoder = Model(latent_inputs, [attr_output, a_output], name='reconstruction')
    # decoder.summary()

    ## 3) instantiate VAE model
    attr_a_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, attr_a_outputs, name='vae')

    return vae