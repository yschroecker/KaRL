import collections
import numpy as np
import tensorflow as tf
import util.ring_buffer

Statistics = collections.namedtuple('Statistics', ['mean_q', 'epsilon'])


class UniformExperienceReplayMemory:
    """
    Simplest experience replay memory. DQN stores new samples in this memory and samples minibatches from this memory.
    In this case the samples are sampled uniformly
    """
    def __init__(self, state_dim, buffer_size=1, mini_batch_size=1, *args, **kwargs):
        """
        Default parmaeters correspond to regular Q learning without memory.
        :param state_dim:
            Dimensionality of the state
        :param buffer_size:
            Number of states to be stored in memory. After buffer_size samples, the oldest samples will be forgotten
        :param mini_batch_size:
            Size of each mini_batch
        :param file_path:
            File to store samples in. Keeps everything in memory if file_path is None (default). Can slow down
            DQN considerably (don't even try without a fast SSD).
        :param dtype:
            Type to store state in. np.float16 by default to save memory. However, this has to be cast to float32
            in DQN for each mini_batch.
        """
        self._state_dim = state_dim
        self._buffer = util.ring_buffer.RingBufferCollection(buffer_size, [self._state_dim, 1, self._state_dim, 1, 1],
                                                             *args, **kwargs)
        self._mini_batch_size = mini_batch_size

    def add_sample(self, state, action, next_state, reward, is_terminal):
        """
        Stores new sample in memory
        """
        self._buffer.append(state, action, next_state, reward, is_terminal)

    def get_mini_batch(self):
        """
        Uniformly samples a minibatch
        :return:
            Tuple of 5 ndarrays. (states, actions, next_States, rewards, is_terminal)
        """
        state, action, next_state, reward, is_terminal = self._buffer.sample(self._mini_batch_size)
        return state, np.atleast_1d(np.squeeze(action)), next_state, np.atleast_1d(np.squeeze(reward)),\
               np.atleast_1d(np.squeeze(is_terminal))

    def batch_size(self):
        """
        :return:
            Size of each mini batch
        """
        return self._mini_batch_size

    @property
    def size(self):
        """
        :return:
            Number of samples stored in this memory
        """
        return self._buffer.size


class DQN:
    """
    DQN algorithm ([Mnih et al., 2015]). 1-step Q-learning with target network & replay memory.
    """
    def __init__(self, network_builder, state_dim, num_actions, optimizer, discount_factor, experience_replay_memory,
                 exploration, update_interval=1, freeze_interval=1, loss_clip_threshold=None, loss_clip_mode='linear',
                 double_dqn=False, create_summaries=False, minimum_memory_size=0,
                 state_preprocessor=lambda x, y=None: x if y is None else (x, y),
                 global_step=tf.get_variable("dqn_step", shape=[], dtype=tf.int32,
                                             initializer=tf.constant_initializer(0), trainable=False)):
        """
        :param network_builder: (state, reuse) -> Q
            Function returning a tensor representing the Q function. May be called multiple times with or without
            parameter sharing. Thus use tf.get_variables instead of tf.Variable and do not change the tf.VariableScope.
            reuse is supplied when reusing parameters but does not need to be used for creating the network. It can,
            however, be useful for creating summaries.
        :param state_dim:
            Dimensionality of the state as a list (not np.ndarray)
        :param num_actions:
            Number of discrete actions. DQN returns an integer between 0 and num_actions to designate the action to be
            taken. Continuous actions are not implemented for DQN at this point.
        :param optimizer:
            Tensorflow optimizer. Original paper uses RMSProp. Adagrad can also be a good choice. Note that the targets
            will change over time and internal state such as the adjusted learning rate and momentum behave differently
            than for supervised learning with fixed training samples.
        :param discount_factor:
            Discount factor of the MDP
        :param experience_replay_memory:
            An instance of an experience replay memory. E.g. dqn.UniformExperienceReplayMemory
        :param exploration:
            Instance of an exploration scheme. E.g. dqn.EpsilonGreedy
        :param update_interval:
            Updates the online Q network every update_interval steps
        :param freeze_interval:
            Updates the target network every freeze_interval steps. Should be multiple of update_interval.
        :param loss_clip_threshold:
            Threshold after which the TD loss is clipped according to loss_clip_mode. None if td loss should not be
            clipped
        :param loss_clip_mode:
            'linear' or 'absolute'. Absolute cuts of loss completely, 'linear' (default) results in a loss that is
            squared up to loss_clip_threshold and linear afterwards
        :param double_dqn:error
            See [Van Hasselt et al., 2015]. If true, chooses actions according to online network as opposed to target
            network when creating target values. (The target values are still calculated w.r.t. the target network)
        :param create_summaries:
            Creates tf summaries. Call add_summaries save the summaries
        :param minimum_memory_size:
            Number of samples to be observed before starting to update the networks.
        :param state_preprocessor: (states, next_states=None) -> (states[, next_states])
            Transformation of states obtained by ReplayMemory. No transformation by default.
        :param global_step:
            Step counter. Uses tf variable 'dqn_step' by default
        """
        self._dqn_step = global_step
        self._build_network = network_builder
        self._num_actions = num_actions
        self._state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="state")
        self._next_state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="state")
        self._action = tf.placeholder(tf.int64, shape=[None], name="action")
        self._reward = tf.placeholder(tf.float32, shape=[None])
        self._target_q_factor = tf.placeholder(tf.float32, shape=[None])
        self._experience_replay_memory = experience_replay_memory
        self._freeze_interval = freeze_interval
        self._update_interval = update_interval
        self._minimum_memory_size = minimum_memory_size
        self._preprocess_states = state_preprocessor
        self._exploration = exploration
        self._samples_since_update = 0

        with tf.variable_scope('online_network'):
            self._q = self._build_network(self._state, False)

        with tf.variable_scope('target_network'):
            next_q_values_target_net = self._build_network(self._next_state, False)

        if double_dqn:
            with tf.variable_scope('target_network', reuse=True):
                next_q_values_q_net = self._build_network(self._next_state, True)
                self._max_next_q = self._array_indexing(next_q_values_target_net, tf.argmax(next_q_values_q_net, 1))
        else:
            self._max_next_q = tf.reduce_max(next_q_values_target_net, 1)

        with tf.variable_scope('target_network', reuse=True):
            q_values = tf.squeeze(self._build_network(self._state, True))
            self._max_action = tf.squeeze(tf.gather(np.arange(self._num_actions), tf.argmax(q_values, 0)))

        q_a = self._array_indexing(self._q, self._action)

        td_error = 2 * (self._reward + discount_factor * self._target_q_factor * self._max_next_q - q_a)
        if loss_clip_threshold is None:
            self._td_loss = tf.reduce_sum(td_error ** 2)
        elif loss_clip_mode == 'linear':
            self._td_loss = tf.reduce_sum(tf.minimum(td_error, loss_clip_threshold) ** 2 +
                                          tf.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self._td_loss = tf.reduce_sum(tf.clip_by_value(td_error, 0, loss_clip_threshold) ** 2)

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
        """
        Save summaries. Only saves after at least one update has been performed.
        :param episode:
            Int denoting the current episode
        """
        if self._last_batch_feed_dict is not None:
            summary_writer.add_summary(tf.get_default_session().run(self._summary_op,
                                                                    feed_dict=self._last_batch_feed_dict), episode)

    def update(self, state, action, next_state, reward, is_terminal):
        """
        Should be called after every step that the agent takes. Performs one iteration of actual learning
        :param state:
            The prior state observed
        :param action:
            The action taken in the given state
        :param next_state:
            The state observed after taking the given action
        :param reward:
            The reward obtained after this transition
        :param is_terminal:
            Whether next_state is a terminal state
        :return: Statistics
            Returns a Statistics object containing diagnostic information
        """
        self._exploration.update()
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

            if self._dqn_step.eval() % self._freeze_interval == 0:
                tf.get_default_session().run(self._copy_weight_ops, feed_dict=feed_dict)

            return Statistics(np.mean(mini_batch_q), self._exploration.epsilon)
        return Statistics(0, self._exploration.epsilon)

    def max_action(self, state):
        """
        Returns the greedy action for evaluation
        """
        return self._max_action.eval(feed_dict={self._state: self._preprocess_states([state])})

    def get_action(self, state):
        """
        Returns the action according to the given exploration scheme
        """
        return self._exploration.get_action(self, state)

    def actions(self):
        """
        Returns an array of all discrete actions
        :return: np.ndarray
            A set of all actions
        """
        return np.arange(self._num_actions)

    def _copy_weights(self):
        ops = []
        with tf.variable_scope('target_network', reuse=True):
            for variable in self._scope_collection('online_network'):
                ops.append(tf.get_variable(variable.name.split('/', 1)[1].split(':', 1)[0]).assign(variable))
        return ops

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

    @staticmethod
    def _array_indexing(tensor, index_tensor):
        one_hot_indices = tf.reshape(tf.one_hot(index_tensor, tf.shape(tensor)[1], 1., 0., axis=-1),
                                     [-1, tf.shape(tensor)[1], 1])
        return tf.squeeze(tf.batch_matmul(tf.reshape(tensor, [-1, 1, tf.shape(tensor)[1]]), one_hot_indices))


class EpsilonGreedy:
    """
    Epsilon greedy exploration scheme. Chooses greedy action with probability 1-epsilon, uniformly random o.w..
    Decays epsilon after each episode
    """
    def __init__(self, initial_epsilon, epsilon_decay=1, min_epsilon=0, decay_type='linear'):
        """
        :param initial_epsilon:
            Initial probability for exploration
        :param epsilon_decay:
            If decay_type='exponential', this is the factor with which epsilon gets multiplied after every iteration.
            If decay_type='linear', this is the fraction of max_epsilon - min_epsilon that gets substracted after each
            iteration.
        :param min_epsilon:
            Final probability for exploration
        :param decay_type:
            Type of decay. See doc for epsilon_decay
        """
        self._initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._decay_type = decay_type

    def get_action(self, learner, state):
        rand = np.random.rand()
        if rand > self.epsilon:
            return learner.max_action(state)
        else:
            return np.random.choice(learner.actions())

    def update(self):
        if self._decay_type == 'exponential':
            self.epsilon = max(self._min_epsilon, self.epsilon * self._epsilon_decay)
        elif self._decay_type == 'linear':
            self.epsilon = max(self._min_epsilon,
                               self.epsilon - self._epsilon_decay * (self._initial_epsilon - self._min_epsilon))
        else:
            assert False, "Invalid epsilon decay type"
