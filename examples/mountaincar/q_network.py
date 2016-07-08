import tensorflow as tf
import numpy as np


class QNetwork:
    def __init__(self):
        self.state_dim = 2
        self.discretized_actions = [0, 1, 2]

    def build_network(self, state):
        input_size = self.state_dim
        hidden1_size = 10
        hidden2_size = 10
        output_size = len(self.discretized_actions)

        hidden1_W = tf.get_variable("hidden1_W", shape=[input_size, hidden1_size],
                                    initializer=tf.truncated_normal_initializer(mean=1, stddev=0.1))
        hidden1_b = tf.get_variable("hidden1_b", shape=[hidden1_size], initializer=tf.constant_initializer(0.1))
        hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(state, hidden1_W), hidden1_b), name="hidden1")

        hidden2_W = tf.get_variable("hidden2_W", shape=[hidden1_size, hidden2_size],
                                    initializer=tf.truncated_normal_initializer(mean=1, stddev=0.1))
        hidden2_b = tf.get_variable("hidden2_b", shape=[hidden2_size], initializer=tf.constant_initializer(0.1))
        hidden2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden1, hidden2_W), hidden2_b), name="hidden2")

        output_W = tf.get_variable("output_W", shape=[hidden2_size, output_size],
                                   initializer=tf.truncated_normal_initializer(mean=1, stddev=0.1))

        output_b = tf.get_variable("output_b", shape=[output_size], initializer=tf.constant_initializer(0.))

        return tf.nn.bias_add(tf.matmul(hidden2, output_W), output_b, name="Q")

