import collections
import numpy as np
import tensorflow as tf
import util.ring_buffer

Statistics = collections.namedtuple('Statistics', ['mean_q'])


class UniformExperienceReplayMemory:
    def __init__(self, state_dim, buffer_size=1, mini_batch_size=1, file_path=None, dtype=np.float16):  # TODO: refactor
        self._state_dim = state_dim
        self._buffer = util.ring_buffer.RingBufferCollection(buffer_size, [self._state_dim, 1, self._state_dim, 1, 1],
                                                             file_path=file_path, dtype=dtype)
        self._mini_batch_size = mini_batch_size

    def add_sample(self, state, action, next_state, reward, target_factor):
        self._buffer.append(state, action, next_state, reward, target_factor)

    def get_mini_batch(self):
        state, action, next_state, reward, target_factor = self._buffer.sample(self._mini_batch_size)
        return state, np.squeeze(action), next_state, np.squeeze(reward), np.squeeze(target_factor)

    def batch_size(self):
        return self._mini_batch_size

    @property
    def size(self):
        return self._buffer.size


class DQN:
    def __init__(self, q_network, optimizer, discount_factor, experience_replay_memory,
                 update_interval=1, freeze_interval=1, loss_clip_threshold=None, loss_clip_mode='linear',
                 double_dqn=False, create_summaries=False, minimum_memory_size=0,
                 state_preprocessor=lambda x, y=None: x if y is None else (x, y),
                 global_step=tf.get_variable("dqn_step", shape=[], dtype=tf.int32,
                                             initializer=tf.constant_initializer(0), trainable=False)):
        self._dqn_step = global_step
        self._q_network = q_network
        self._state = tf.placeholder(tf.float32, shape=[None] +  q_network.state_dim, name="state")
        self._next_state = tf.placeholder(tf.float32, shape=[None] + q_network.state_dim, name="state")
        self._action = tf.placeholder(tf.int64, shape=[None], name="action")
        self._reward = tf.placeholder(tf.float32, shape=[None])
        self._target_q_factor = tf.placeholder(tf.float32, shape=[None])
        self._experience_replay_memory = experience_replay_memory
        self._freeze_interval = freeze_interval
        self._update_interval = update_interval
        self._minimum_memory_size = minimum_memory_size
        self._preprocess_states = state_preprocessor
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

        self._td_loss = self._reward + discount_factor * self._target_q_factor * self._max_next_q - q_a
        if loss_clip_threshold is None:
            self._td_loss = tf.reduce_sum(self._td_loss ** 2)
        elif loss_clip_mode == 'linear':
            self._td_loss = tf.reduce_sum(tf.minimum(self._td_loss, loss_clip_threshold) ** 2 +
                                          tf.maximum(self._td_loss - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self._td_loss = tf.reduce_sum(tf.clip_by_value(self._td_loss, 0, loss_clip_threshold) ** 2)

        self._td_loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'online_network'))

        self._optimizer = optimizer

        td_gradient = self._optimizer.compute_gradients(self._td_loss, self._scope_collection('online_network'))
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
            tf.scalar_summary("sample-R", tf.reduce_mean(self._reward))
            tf.scalar_summary("q", tf.reduce_mean(self._q))
            tf.scalar_summary("td loss", self._td_loss)
            self._summary_op = tf.merge_all_summaries()

    def add_summaries(self, summary_writer, episode):
        if self._last_batch_feed_dict is not None:
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

    def update(self, state, action, next_state, reward, is_terminal):
        self._experience_replay_memory.add_sample(state, action, next_state, reward, 0 if is_terminal else 1)

        self._samples_since_update += 1
        if self._samples_since_update >= self._update_interval and \
                        self._experience_replay_memory.size >= self._minimum_memory_size:
            self._samples_since_update = 0
            states, actions, next_states, rewards, target_q_factor = self._experience_replay_memory.get_mini_batch()
            transformed_states, transformed_next_states = self._preprocess_states(states, next_states)
            feed_dict = {self._state: transformed_states, self._action: actions,
                         self._next_state: transformed_next_states, self._reward: rewards,
                         self._target_q_factor: target_q_factor}
            self._last_batch_feed_dict = feed_dict

            # Profiling
            # import sys
            # from tensorflow.python.client import timeline
            # run_metadata = tf.RunMetadata()
            # tf.get_default_session().run([self._update_op] + self._copy_weight_ops, feed_dict=feed_dict,
            #                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            #                              run_metadata=run_metadata)
            # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            # with open('/home/yannick/scpout/timeline.json', 'w') as f:
            #     f.write(trace.generate_chrome_trace_format())
            # sys.exit(0)

            mini_batch_q, td_loss, _ = tf.get_default_session().run([self._q, self._td_loss, self._update_op],
                                                                    feed_dict=feed_dict)

            if td_loss > 10**3:
                import pdb;pdb.set_trace()

            if self._dqn_step.eval() % self._freeze_interval == 0:
                tf.get_default_session().run(self._copy_weight_ops, feed_dict=feed_dict)

            return Statistics(np.mean(mini_batch_q))
        return Statistics(0)

    def max_action(self, state):
        return self._max_action.eval(feed_dict={self._state: self._preprocess_states([state])})

    def actions(self):
        return self._q_network.discretized_actions

    @staticmethod
    def _array_indexing(tensor, index_tensor):
        one_hot_indices = tf.reshape(tf.one_hot(index_tensor, tf.shape(tensor)[1], 1., 0., axis=-1),
                                     [-1, tf.shape(tensor)[1], 1])
        return tf.squeeze(tf.batch_matmul(tf.reshape(tensor, [-1, 1, tf.shape(tensor)[1]]), one_hot_indices))


class EpsilonGreedy:
    def __init__(self, learner, initial_epsilon, epsilon_decay=1, min_epsilon=0, decay_type='linear'):
        self._learner = learner
        self._initial_epsilon = initial_epsilon
        self._epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._decay_type = decay_type

    def get_action(self, state):
        rand = np.random.rand()
        if rand > self._epsilon:
            return self._learner.max_action(state)
        else:
            return np.random.choice(self._learner.actions())

    def update(self):
        if self._decay_type == 'exponential':
            self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
        elif self._decay_type == 'linear':
            self._epsilon = max(self._min_epsilon,
                                self._epsilon - self._epsilon_decay * (self._initial_epsilon - self._min_epsilon))
        else:
            assert False, "Invalid epsilon decay type"
