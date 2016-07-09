import numpy as np
import tensorflow as tf


class UniformExperienceReplayMemory:
    class RingBuffer:
        def __init__(self, capacity, element_dim):
            self._ringbuffer = np.zeros([capacity, element_dim])
            self._head = 0
            self._size = 0

        def append(self, elem):
            self._ringbuffer[self._head, :] = elem
            self._head = (self._head + 1) % self._ringbuffer.shape[0]
            self._size = min(self._size + 1, self._ringbuffer.shape[0])

        def sample(self, num_samples):
            indices = np.random.choice(self._size, num_samples)
            return self._ringbuffer[indices, :]

    def __init__(self, state_dim, buffer_size=1, mini_batch_size=1):  # TODO: refactor
        self._state_dim = state_dim
        self._buffer = self.RingBuffer(buffer_size, 2 + 2 * state_dim)
        self._mini_batch_size = mini_batch_size

    def add_sample(self, state, action, next_state, reward):
        self._buffer.append(np.hstack([np.atleast_1d(state), [action], np.atleast_1d(next_state), [reward]]))

    def get_mini_batch(self):
        mini_batch = self._buffer.sample(self._mini_batch_size)
        return mini_batch[:, :self._state_dim], mini_batch[:, self._state_dim], \
               mini_batch[:, self._state_dim + 1: -1], mini_batch[:, -1]

    def batch_size(self):
        return self._mini_batch_size


class DQN:
    def __init__(self, q_network, optimizer, discount_factor, experience_replay_memory,
                 update_interval=1, freeze_interval=1, loss_clip_threshold=None, loss_clip_mode='linear',
                 double_dqn=False, create_summaries=False,
                 global_step=tf.get_variable("dqn_step", shape=[], dtype=tf.int32,
                                             initializer=tf.constant_initializer(0), trainable=False)):
        self._dqn_step = global_step
        self._q_network = q_network
        self._state = tf.placeholder(tf.float32, shape=[None, q_network.state_dim], name="state")
        self._next_state = tf.placeholder(tf.float32, shape=[None, q_network.state_dim], name="state")
        self._action = tf.placeholder(tf.int64, shape=[None], name="action")
        self._reward = tf.placeholder(tf.float32, shape=[None])
        self._experience_replay_memory = experience_replay_memory
        self._freeze_interval = freeze_interval
        self._update_interval = update_interval
        self._samples_since_update = 0

        with tf.variable_scope('online_network'):
            self._q = self._q_network.build_network(self._state, False)

        with tf.variable_scope('target_network'):
            next_q_values_target_net = self._q_network.build_network(self._next_state, False)

        if double_dqn:
            with tf.variable_scope('target_network', reuse=True):
                next_q_values_q_net = self._q_network.build_network(self._next_state, True)
                self._max_next_q = self._array_indexing(next_q_values_target_net, tf.argmax(next_q_values_q_net, 1))
        else:
            self._max_next_q = tf.reduce_max(next_q_values_target_net, 1)

        with tf.variable_scope('target_network', reuse=True):
            q_values = tf.squeeze(self._q_network.build_network(self._state, True))
            self._max_action = tf.squeeze(tf.gather(self._q_network.discretized_actions, tf.argmax(q_values, 0)))

        q_a = self._array_indexing(self._q, self._action)

        td_loss = self._reward + discount_factor * self._max_next_q - q_a
        if loss_clip_threshold is None:
            td_loss = tf.reduce_mean(td_loss ** 2)
        elif loss_clip_mode == 'linear':
            td_loss = tf.reduce_mean(tf.minimum(td_loss, loss_clip_threshold) ** 2 + \
                                     tf.maximum(td_loss - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            td_loss = tf.reduce_mean(tf.clip_by_value(td_loss, 0, loss_clip_threshold) ** 2)

        td_loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'online_network'))

        self._optimizer = optimizer

        td_gradient = self._optimizer.compute_gradients(td_loss, self._scope_collection('online_network'))
        self._update_op = self._optimizer.apply_gradients(td_gradient, global_step=self._dqn_step)
        self._copy_weight_ops = self._copy_weights()

        self._last_batch_feed_dict = None
        if create_summaries:
            for variable in self._scope_collection('online_network'):
                tf.histogram_summary(variable.name, variable)
            for variable in self._scope_collection('target_network'):
                tf.histogram_summary(variable.name, variable)
            for gradient, variable in td_gradient:
                tf.histogram_summary(variable.name + "_gradient", gradient)
            self._summary_op = tf.merge_all_summaries()

    def add_summaries(self, summary_writer, episode):
        summary_writer.add_summary(tf.get_default_session().run(self._summary_op,
                                                                feed_dict=self._last_batch_feed_dict), episode)

    @staticmethod
    def _print_gradient(update_op, gradient):
        ops = []
        for grad, var in gradient:
            ops.append(tf.Print(grad, [grad], summarize=1000))
        with tf.control_dependencies(ops + [update_op]):
            return tf.no_op()

    @staticmethod
    def _scope_collection(scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def _copy_weights(self):
        ops = []
        with tf.variable_scope('target_network', reuse=True):
            for variable in self._scope_collection('online_network'):
                ops.append(tf.get_variable(variable.name.split('/', 1)[1].split(':', 1)[0]).assign(variable))
        return ops

    def update(self, state, action, next_state, reward):
        self._experience_replay_memory.add_sample(state, action, next_state, reward)

        self._samples_since_update += 1
        if self._samples_since_update == self._update_interval:
            self._samples_since_update = 0
            states, actions, next_states, rewards = self._experience_replay_memory.get_mini_batch()
            feed_dict = {self._state: states, self._action: actions,
                         self._next_state: next_states, self._reward: rewards}
            self._last_batch_feed_dict = feed_dict

            tf.get_default_session().run(self._update_op, feed_dict=feed_dict)

            if self._dqn_step.eval() % self._freeze_interval == 0:
                tf.get_default_session().run(self._copy_weight_ops, feed_dict=feed_dict)

    def max_action(self, state):
        return self._max_action.eval(feed_dict={self._state: [state]})

    def actions(self):
        return self._q_network.discretized_actions

    @staticmethod
    def _array_indexing(tensor, index_tensor):
        one_hot_indices = tf.reshape(tf.one_hot(index_tensor, tf.shape(tensor)[1], 1., 0., axis=-1),
                                     [-1, tf.shape(tensor)[1], 1])
        return tf.squeeze(tf.batch_matmul(tf.reshape(tensor, [-1, 1, tf.shape(tensor)[1]]), one_hot_indices))


class EpsilonGreedy:
    def __init__(self, learner, initial_epsilon, epsilon_decay=1, min_epsilon=0):
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
