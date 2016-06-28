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
            indices = np.random.choice(np.arange(self._size), num_samples)
            return self._ringbuffer[indices, :]

    def __init__(self, buffer_size=1, mini_batch_size=1):
        self._buffer = self.RingBuffer(buffer_size, 4)  # not a deque to avoid casting between lists and numpy arrays
        self._mini_batch_size = mini_batch_size

    def add_sample(self, state, action, next_state, reward):
        self._buffer.append([state, action, next_state, reward])

    def get_mini_batch(self):
        return self._buffer.sample(self._mini_batch_size)


class DQN:
    def __init__(self, q_network, learning_rate, discount_factor, experience_replay_memory, freeze_interval):
        self._dqn_step = tf.get_variable("dqn_step", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0))
        self._q_network = q_network
        self._state = tf.placeholder(tf.float32, shape=[None] + q_network.state_dim, name="state")
        self._next_state = tf.placeholder(tf.float32, shape=[None] + q_network.state_dim, name="state")
        self._action = tf.placeholder(tf.float32, shape=[None] + q_network.action_dim, name="action")
        self._reward = tf.placeholder(tf.float32, shape=[None])
        self._experience_replay_memory = experience_replay_memory
        self._freeze_interval = freeze_interval

        with tf.variable_scope('online_network'):
            self._q = self._q_network.build_network(self._state, self._action)

        with tf.variable_scope('target_network') as scope:
            next_q_values = []
            for action in self._q_network.discretized_actions:
                next_q_values.append(self._q_network.build_network(self._next_state, tf.constant(action,
                                                                                                 dtype=tf.float32)))
                scope.reuse_variables()

            self._max_next_q = tf.reduce_max(tf.pack(next_q_values), 0)

        with tf.variable_scope('target_network', reuse=True):
            q_values = tf.pack([self._q_network.build_network(self._state, tf.constant(action, dtype=tf.float32))
                                for action in self._q_network.discretized_actions])
            self._max_action = tf.squeeze(tf.gather(self._q_network.discretized_actions, tf.argmax(q_values, 0)))

        td_error = 0.5 * tf.reduce_mean((self._reward + discount_factor * self._max_next_q - self._q)**2)
        tf.histogram_summary("td error", td_error)

        self._optimizer = tf.train.GradientDescentOptimizer(tf.cast(learning_rate, tf.float32))

        td_gradient = self._optimizer.compute_gradients(td_error, self._scope_collection('online_network'))
        self._update_op = self._optimizer.apply_gradients(td_gradient, global_step=self._dqn_step)

    @staticmethod
    def _scope_collection(scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def _copy_weights(self):
        ops = []
        with tf.variable_scope('target_network', reuse=True):
            for variable in self._scope_collection('online_network'):
                ops.append(tf.get_variable(variable.name.split('/', 1)[1].split(':', 1)[0]).assign(variable))
        return ops

    def update(self, state, action, reward, next_state):
        self._experience_replay_memory.add_sample(state, action, next_state, reward)
        mini_batch = self._experience_replay_memory.get_mini_batch()
        feed_dict = {self._state: mini_batch[:, 0], self._action: mini_batch[:, 1],
                     self._next_state: mini_batch[:, 2], self._reward: mini_batch[:, 3]}

        if self._dqn_step.eval() % self._freeze_interval == 0:
            tf.get_default_session().run(self._copy_weights(), feed_dict=feed_dict)

        tf.get_default_session().run(self._update_op, feed_dict=feed_dict)

    def max_action(self, state):
        return self._max_action.eval(feed_dict={self._state: [state]})

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
