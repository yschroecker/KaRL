import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, q_network, learning_rate, discount_factor):
        self._q_network = q_network
        self._state = tf.placeholder(tf.float32, shape=q_network.state_dim, name="state")
        self._next_state = tf.placeholder(tf.float32, shape=q_network.state_dim, name="state")
        self._action = tf.placeholder(tf.float32, shape=q_network.action_dim, name="action")
        self._reward = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope("q_network"):
            self._q = self._q_network.build_network(self._state, self._action)

        with tf.variable_scope("q_network", reuse=True):
            q_values = tf.pack([self._q_network.build_network(self._state, tf.constant(action, dtype=tf.float32))
                                for action in self._q_network.discretized_actions])
            self._max_action = tf.squeeze(tf.gather(self._q_network.discretized_actions, tf.argmax(q_values, 0)))

        with tf.variable_scope("q_network", reuse=True):
            next_q_values = tf.pack([self._q_network.build_network(self._next_state, tf.constant(action, dtype=tf.float32))
                                     for action in self._q_network.discretized_actions])
            self._max_next_q = tf.reduce_max(next_q_values, 0)

        td_error = self._reward + discount_factor * self._max_next_q - self._q

        self._optimizer = tf.train.GradientDescentOptimizer(tf.cast(learning_rate, tf.float32))
        td_gradient = self._optimizer.compute_gradients(td_error)
        self._update_op = self._optimizer.apply_gradients(td_gradient)

    def update(self, state, action, reward, next_state):
        tf.get_default_session().run(self._update_op, feed_dict={self._state: state, self._action: action,
                                                                 self._next_state: next_state, self._reward: reward})

    def max_action(self, state):
        return self._max_action.eval(feed_dict={self._state: state})

    def actions(self):
        return self._q_network.discretized_actions


class EpsilonGreedy:
    def __init__(self, learner, initial_epsilon, epsilon_decay, min_epsilon):
        self._learner = learner
        self._initial_epsilon = initial_epsilon
        self._epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon

    def get_action(self, state):
        rand = np.random.rand()
        if rand > self._epsilon:
            return self._learner.max_action(state)
        else:
            return np.random.choice(self._learner.actions())

    def update(self):
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
