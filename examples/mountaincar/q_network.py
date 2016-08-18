import tensorflow as tf
import tflearn
import numpy as np


def build_q_network(state, reuse):
    state = tf.reshape(state, shape=(-1, 2))
    hidden1 = tflearn.fully_connected(state, 40, activation='elu', weights_init='uniform_scaling',
                                      bias_init=tf.constant(0., shape=(40,)),
                                      name="hidden1", reuse=None)
    hidden2 = tflearn.fully_connected(hidden1, 40, activation='elu', weights_init='uniform_scaling',
                                      bias_init=tf.constant(0., shape=(40,)),
                                      name="hidden2", reuse=None)
    hidden3 = tflearn.fully_connected(hidden2, 40, activation='elu', weights_init='uniform_scaling',
                                      bias_init=tf.constant(0., shape=(40,)),
                                      name="hidden3", reuse=None)
    hidden4 = tflearn.fully_connected(hidden3, 40, activation='elu', weights_init='uniform_scaling',
                                      bias_init=tf.constant(0., shape=(40,)),
                                      name="hidden4", reuse=None)
    output = tflearn.fully_connected(hidden4, 3, weights_init='uniform_scaling', bias=False,
                                     name="output", reuse=None)
    return output

