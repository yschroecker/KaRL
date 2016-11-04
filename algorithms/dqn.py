import collections
import pickle

import numpy as np

import algorithms.theano_backend.temporal_difference as td
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

    @staticmethod
    def _save_path(dir_path, name):
        return '%s/replay_memory_%s' % (dir_path, name)

    def save(self, dir_path, name):
        with open(self._save_path(dir_path, name), 'wb') as f:
            pickle.dump([self._state_dim, self._mini_batch_size], f)
        self._buffer.save(dir_path, name)

    @classmethod
    def load(cls, dir_path, name):
        with open(cls._save_path(dir_path, name), 'rb') as f:
            [state_dim, mini_batch_size] = pickle.load(f)
        uniform_experience_replay_memory = cls.__new__(cls)
        uniform_experience_replay_memory._state_dim = state_dim
        uniform_experience_replay_memory._mini_batch_size = mini_batch_size
        uniform_experience_replay_memory._buffer = util.ring_buffer.RingBufferCollection.load(dir_path, name)
        return uniform_experience_replay_memory

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


def no_preprocessor(x, y=None):
    return x if y is None else (x, y)


class DQN:
    """
    DQN algorithm ([Mnih et al., 2015]). 1-step Q-learning with target network & replay memory.
    """
    def __init__(self, network_builder, state_dim, num_actions, optimizer, discount_factor, experience_replay_memory,
                 exploration, update_interval=1, freeze_interval=1, loss_clip_threshold=None, loss_clip_mode='linear',
                 td_rule='q-learning', create_summaries=True, minimum_memory_size=0,
                 state_preprocessor=no_preprocessor):
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
        """
        self._update_counter = 0
        self._num_actions = num_actions
        self._freeze_interval = freeze_interval
        self._update_interval = update_interval
        self._minimum_memory_size = minimum_memory_size
        self._preprocess_states = state_preprocessor
        self._exploration = exploration

        self._td_learner = td.TemporalDifferenceLearnerQ(
            network_builder=network_builder, optimizer=optimizer, state_dim=state_dim, num_actions=num_actions,
            discount_factor=discount_factor, td_rule=td_rule, loss_clip_threshold=loss_clip_threshold,
            loss_clip_mode=loss_clip_mode, create_summaries=create_summaries)

        self._last_batch_feed_dict = None
        self._samples_since_update = 0
        self._experience_replay_memory = experience_replay_memory

    @staticmethod
    def _save_path(dir_path, name):
        return '%s/dqn_%s' % (dir_path, name)

    def save(self, dir_path, name):
        with open(self._save_path(dir_path, name + "_exp"), 'wb') as f:
            pickle.dump(self._samples_since_update, f)
        with open(self._save_path(dir_path, name + "_net"), 'wb') as f:
            pickle.dump(self._td_learner, f)
        self._experience_replay_memory.save(dir_path, name)

    def load(self, dir_path, name, experience_memory_type=UniformExperienceReplayMemory):
        with open(self._save_path(dir_path, name + "_exp"), 'rb') as f:
            self._samples_since_update = pickle.load(f)
        with open(self._save_path(dir_path, name + "_net"), 'rb') as f:
            self._td_learner = pickle.load(f)
        self._experience_replay_memory = experience_memory_type.load(dir_path, name)

    def add_summaries(self, summary_writer, episode):
        """
        Save summaries. Only saves after at least one update has been performed.
        :param episode:
            Int denoting the current episode
        """
        self._td_learner.add_summaries(summary_writer, episode)

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
            self._update_counter += 1
            states, actions, next_states, rewards, target_q_factor = self._experience_replay_memory.get_mini_batch()
            transformed_states, transformed_next_states = self._preprocess_states(states, next_states)

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

            mini_batch_q, td_loss = self._td_learner.bellman_operator_update(transformed_states, actions, rewards,
                                                                             transformed_next_states, target_q_factor)

            if self._update_counter >= self._freeze_interval:
                self._update_counter = 0
                self._td_learner.fixpoint_update()

            return Statistics(np.mean(mini_batch_q), self._exploration.epsilon)
        return Statistics(0, self._exploration.epsilon)

    def max_action(self, state):
        """
        Returns the greedy action for evaluation
        """
        return self._td_learner.max_action(self._preprocess_states([state]))[0]

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
