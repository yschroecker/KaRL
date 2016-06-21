import tensorflow as tf
import numpy as np

class QNetwork:
    def build_network(self, state, action):
        input_size = self.action_dim + self.state_dim
        hidden1_size = 10
        hidden2_size = 10
        output_size = 1

        state_action = tf.reshape(tf.concat(0, [state, action]), shape=[1, -1], name="state_action")
        hidden1_W = tf.get_variable("hidden1_W", shape=[input_size, hidden1_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        hidden1_b = tf.get_variable("hidden1_b", shape=[hidden1_size], initializer=tf.constant_initializer(0.1))
        hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(state_action, hidden1_W), hidden1_b), name="hidden1")

        hidden2_W = tf.get_variable("hidden2_W", shape=[hidden1_size, hidden2_size],
                                    initializer=tf.truncated_normal_initializer(0.1))
        hidden2_b = tf.get_variable("hidden2_b", shape=[hidden2_size], initializer=tf.constant_initializer(0.1))
        hidden2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden1, hidden2_W), hidden2_b), name="hidden2")

        output_W = tf.get_variable("output_W", shape=[hidden2_size, output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))

        output_b = tf.get_variable("output_b", shape=[output_size], initializer=tf.constant_initializer(0.))

        return tf.nn.bias_add(tf.matmul(hidden2, output_W), output_b, name="Q")

    def __init__(self):
        self.action_dim = 1
        self.state_dim = 2
        self.discretized_actions = [0, 1, 2]

        #self.state = tf.placeholder(tf.float32, shape=[self.state_dim], name="state")
        #self.action = tf.placeholder(tf.float32, shape=[self.action_dim], name="action")

        #with tf.variable_scope("q_network"):
            #self._q = self._build_network(self.state, self.action)
        #with tf.variable_scope("q_network", reuse=True):
            #q_values = tf.pack([self._build_network(self.state, tf.constant([action], dtype=tf.float32))
                                #for action in self.discretized_actions])
            #self._max_q = tf.reduce_max(q_values, 0)
        #self.network_gradients = tf.gradients(self._Q, state_action, name="gradients")

    #def q(self, state, action):
        #return self._q.eval(feed_dict={self.state: state, self.action: np.atleast_1d(action)})

    #def max_action(self, state):
        #q_values = [self.q(state, action) for action in self.discretized_actions]
        #return self.discretized_actions[np.argmax(q_values)]

    #def max_q(self, state):
        #q_values = [self.q(state, action) for action in self.discretized_actions]
        #return max(q_values)





