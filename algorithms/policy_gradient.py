import tensorflow as tf
import numpy as np
import util.tensor
import util.debug
import algorithms.temporal_difference as td
import uuid
import pickle


class DiscretePolicy:
    def __init__(self, state_dim, num_actions, policy_network_builder, _=None, suffix=None):
        self._build_policy_network = policy_network_builder
        self._num_actions = num_actions
        self._state_dim = state_dim
        self._states = tf.placeholder(tf.float32, shape=[None] + state_dim, name="states")
        if suffix is None:
            suffix = uuid.uuid4()
        self._scope = 'policy_network_%s' % suffix
        self._init_load_path = None

    def init(self, policy_builder_args={}):
        with tf.variable_scope(self._scope, reuse=None):
            self._policy = self._build_policy_network(self._states, **policy_builder_args)
        self.variables = util.tensor.scope_collection(self._scope, tf.GraphKeys.TRAINABLE_VARIABLES)
        self._saver = tf.train.Saver(self.variables)
        if self._init_load_path is not None:
            self._saver.restore(tf.get_default_session(), self._init_load_path)

    def __call__(self, state, action, policy_builder_args={}):
        with tf.variable_scope(self._scope, reuse=True):
            policy = self._build_policy_network(state, **policy_builder_args)
            policy_action = util.tensor.index(policy, action)
        return policy_action

    def log(self, state, action):
        return tf.log(self(state, action))

    def sample(self, state):
        probabilities = self.state_probabilities(state)
        return np.random.choice(np.arange(self._num_actions), p=probabilities)

    def greedy(self, state):
        probabilities = self.state_probabilities(state)
        return np.argmax(probabilities)

    def state_probabilities(self, state, feed_dict={}):
        feed_dict.update({self._states: [state]})
        return self._policy.eval(feed_dict=feed_dict)[0]

    def save(self, path):
        self._saver.save(tf.get_default_session(), '%s/policy_variables' % path)
        with open('%s/policy' % path, 'wb') as f:
            pickle.dump([self._state_dim, self._num_actions, self._build_policy_network, self._scope], f)

    @classmethod
    def load(cls, path):
        with open('%s/policy' % path, 'rb') as f:
            data = pickle.load(f)
        policy = cls(data[0], data[1], data[2], None)
        policy._scope = data[3]
        policy._init_load_path = '%s/policy_variables' % path
        return policy


class AdvantageActorCriticBase:
    def __init__(self, state_dim, policy, value_network_builder, actor_optimizer, critic_optimizer,
                 discount_factor, loss_clip_threshold=1, loss_clip_mode='linear', create_summaries=True,
                 td_rule='deepmind-n-step',
                 global_step=tf.get_variable("ac_step", shape=[], dtype=tf.int32,
                                             initializer=tf.constant_initializer(0), trainable=False),
                 steps_per_update=30, policy_call_arguments={}):

        self._policy = policy
        self._policy.init()

        self._empty_observervations()

        self._state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="state")
        self._action = tf.placeholder(tf.int32, shape=[None], name="action")
        self._next_state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="next_state")
        self._reward = tf.placeholder(tf.float32, shape=[None], name="reward")
        self._target_factor = tf.placeholder(tf.float32, shape=[None])  # 0 for terminal states.
        self._steps_per_update = steps_per_update
        self._steps_since_update = 0

        self._td_learner = td.TemporalDifferenceLearnerV(value_network_builder, critic_optimizer, discount_factor,
                                                         loss_clip_threshold=loss_clip_threshold,
                                                         loss_clip_mode=loss_clip_mode, td_rule=td_rule,
                                                         create_summaries=create_summaries, global_step=global_step,
                                                         state=self._state, reward=self._reward,
                                                         next_state=self._next_state, target_factor=self._target_factor)

        self._advantages = self._reward + discount_factor * self._td_learner.next_v - self._td_learner.v
        use_natural = True
        if use_natural:
            log_policy = lambda state, action: policy.log(state, action, **policy_call_arguments)
            log_policy_gradient = util.tensor.GradientVector(
                util.tensor.vector_gradient(log_policy, [self._state, self._action],
                                            steps_per_update, actor_optimizer), True)
            features = log_policy_gradient.flattened_gradient
            q_weights = tf.squeeze(util.tensor.least_squares(features, self._advantages, method='matrix_solve',
                                                             regularize=0.001))
            self._actor_gradients = log_policy_gradient.new(-q_weights[:log_policy_gradient.flattened_gradient.get_shape()[0].value], False).reshaped_gradient
        else:
            self._actor_gradients = actor_optimizer.compute_gradients(tf.reduce_mean(
                -policy.log(self._state, self._action, **policy_call_arguments) * self._advantages),
                var_list=policy.variables)
        self._critic_gradients = self._td_learner.td_gradient
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

    def get_action(self, state):
        return self._policy.sample(state)

    def needs_update(self, is_terminal):
        self._steps_since_update += 1
        if is_terminal or self._steps_since_update >= self._steps_per_update:
            self._steps_since_update = 0
            return True
        return False

    def _empty_observervations(self):
        self._observed_states = []
        self._observed_actions = []
        self._observed_next_states = []
        self._observed_rewards = []
        self._observed_target_factors = []


class AdvantageActorCritic(AdvantageActorCriticBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actor_update = self._actor_optimizer.apply_gradients(self._actor_gradients)
        self._critic_update = self._critic_optimizer.apply_gradients(self._critic_gradients)

    def update(self, state, action, next_state, reward, is_terminal, feed_dict={}):
        self._observed_states.append(state)
        self._observed_actions.append(action)
        self._observed_next_states.append(next_state)
        self._observed_rewards.append(reward)
        self._observed_target_factors.append(0 if is_terminal else 1)

        self._steps_since_update += 1
        if self.needs_update(is_terminal):
            feed_dict.update({self._state: self._observed_states,
                              self._action: self._observed_actions,
                              self._next_state: self._observed_next_states,
                              self._reward: self._observed_rewards,
                              self._target_factor: self._observed_target_factors})

            self._actor_update.run(feed_dict=feed_dict)
            self._critic_update.run(feed_dict=feed_dict)
            tf.get_default_session().run(self._td_learner.copy_weights_ops, feed_dict=feed_dict)

            self._steps_since_update = 0
            self._empty_observervations()


class EpisodicPG:
    def __init__(self, num_sample_episodes):
        self._num_sample_episodes = num_sample_episodes
        self._episode_counter = 0

    def update(self, state, action, next_state, reward, is_terminal, feed_dict={}):
        if is_terminal:
            self.new_episode(feed_dict)
            if self.needs_update(True):
                self.episodic_update(feed_dict)
                self._episode_counter = 0
            else:
                self._episode_counter += 1

    def needs_update(self, is_terminal):
        return is_terminal and self._episode_counter + 1 >= self._num_sample_episodes


class REINFORCE(EpisodicPG):
    def __init__(self, state_dim, policy, optimizer, discount_factor, num_sample_episodes=3, baseline='constant',
                 policy_call_arguments=None):
        super().__init__(num_sample_episodes)

        if policy_call_arguments is None:
            policy_call_arguments = [{}] * num_sample_episodes

        self._states = [[]]
        self._actions = [[]]
        self._qs = [[]]

        self._episodes_states = [tf.placeholder(tf.float32, shape=[None] + state_dim)
                                 for _ in range(num_sample_episodes)]
        self._episodes_actions = [tf.placeholder(tf.int32, shape=[None]) for _ in range(num_sample_episodes)]
        self._episodes_qs = [tf.placeholder(tf.float32, shape=[None]) for _ in range(num_sample_episodes)]
        self._policy = policy
        self._policy.init()
        self._discount_factor = discount_factor
        self._episode_t = 0
        self._num_sample_episodes = num_sample_episodes

        # REINFORCE gradient

        #stepwise REINFORCE gradient:
        # \sum_{i=1}^N \sum_{t=1}^{T} (\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \sum_{h=t}^T r_h )
        # grad = E[SUM_t(dlogp_t Q)]
        #Optimal baseline
        # b = E[SUM_t(dlogp_t) SUM_t(dlogp_t Q)]/E[SUM_t(dlogp_t)²]
        # grad = E[SUM_t(dlogp_t (Q - b))]
        log_policies = [policy.log(states, actions, **episode_policy_call_arguments)
                        for states, actions, episode_policy_call_arguments in
                        zip(self._episodes_states, self._episodes_actions, policy_call_arguments)]
        episode_policy_gradients = [util.tensor.GradientWrapper(optimizer.compute_gradients(-log_policy * qs))
                                    for log_policy, qs in zip(log_policies, self._episodes_qs)]
        if baseline == 'constant':
            episode_inner_gradients = [util.tensor.GradientWrapper(optimizer.compute_gradients(log_policy))
                                       for log_policy, qs in zip(log_policies, self._episodes_qs)]
            e_q_dlogpi_dlogpi = sum([episode_policy_gradient * episode_inner_gradient
                                     for episode_policy_gradient, episode_inner_gradient
                                     in zip(episode_policy_gradients, episode_inner_gradients)])
            e_dlogpi_dlogpi = sum([episode_inner_gradient * episode_inner_gradient
                                   for episode_inner_gradient
                                   in episode_inner_gradients])
            b = e_q_dlogpi_dlogpi / e_dlogpi_dlogpi.nonzero()

            b_dlogpi = [b * episode_inner_gradient for episode_inner_gradient in episode_inner_gradients]
            e_p_g_with_baseline = [(episode_policy_gradient - baseline)
                                   for episode_policy_gradient, baseline
                                   in zip(episode_policy_gradients, b_dlogpi)]
        # elif baseline == 'time-varying': # (TODO: time-varying baseline)
        #     #logpi_t = [[policy.log_policy([state], [action]) for state, action in zip(states, actions)]
        #                #for states, actions in zip(self._episodes_states, self._episodes_actions)]
        #     dlogpi_t = [tf.map_fn(lambda x: [grad_i[0] for grad_i in optimizer.compute_gradients(x)], log_policy)
        #                 for log_policy in log_policies]
        #     dlogpi_t_q = [dlogpi_t_i * q for dlogpi_t_i, q in zip(dlogpi_t, self._episodes_qs)]
        #
        #
        #     e_p_g_with_baseline = episode_policy_gradients
        elif baseline == 'none':
            e_p_g_with_baseline = episode_policy_gradients
        else:
            assert False, "Invalid Baseline"

        self._policy_gradient = (sum(e_p_g_with_baseline)/num_sample_episodes).get()

        self._update_op = optimizer.apply_gradients(self._policy_gradient)
        # self._update_op = util.debug.print_gradient(self._update_op, self._policy_gradient, message="actor gradient")

    def update(self, state, action, next_state, reward, is_terminal, feed_dict={}):
        self._episode_t += 1

        self._states[-1].append(state)
        self._actions[-1].append(action)
        for t in range(self._episode_t - 1):
            self._qs[-1][t] += self._discount_factor ** (self._episode_t - t) * reward
        self._qs[-1].append(reward)
        super().update(state, action, next_state, reward, is_terminal, feed_dict)

    def new_episode(self, feed_dict):
        self._states.append([])
        self._actions.append([])
        self._qs.append([])
        self._episode_t = 0

    def episodic_update(self, feed_dict):
        for episode_qs_tensor, episode_qs in zip(self._episodes_qs, self._qs):
            feed_dict[episode_qs_tensor] = episode_qs
        for episode_states_tensor, episode_states in zip(self._episodes_states, self._states):
            feed_dict[episode_states_tensor] = episode_states
        for episode_actions_tensor, episode_actions in zip(self._episodes_actions, self._actions):
            feed_dict[episode_actions_tensor] = episode_actions
        self._update_op.run(feed_dict=feed_dict)

        self._states = [[]]
        self._actions = [[]]
        self._qs = [[]]

    def get_action(self, state):
        return self._policy.sample(state)


class eNAC(EpisodicPG):
    def __init__(self, state_dim, policy, optimizer, discount_factor, num_sample_episodes=3, l2_regularization=0.3,
                 policy_call_arguments=None):
        super().__init__(num_sample_episodes)

        if policy_call_arguments is None:
            policy_call_arguments = [{}] * num_sample_episodes

        self._states = [[]]
        self._actions = [[]]
        self._Rs = [0]

        self._episodes_states = [tf.placeholder(tf.float32, shape=[None] + state_dim, name="episode_states_%d" % i)
                                 for i in range(num_sample_episodes)]
        self._episodes_actions = [tf.placeholder(tf.int32, shape=[None], name="episode_actions_%d" % i)
                                  for i in range(num_sample_episodes)]
        self._episode_discount_weights = [tf.placeholder(tf.float32, shape=[None], name="discount_weights_%d" % i)
                                          for i in range(num_sample_episodes)]
        self._episodes_Rs = tf.placeholder(tf.float32, shape=[num_sample_episodes], name="R")
        self._policy = policy
        self._policy.init()
        self._discount_factor = discount_factor
        self._episode_t = 0
        self._num_sample_episodes = num_sample_episodes

        log_gradients = [util.tensor.GradientVector(optimizer.compute_gradients(tf.reduce_sum(
            episode_discount_weights * policy.log(states, actions, **episode_policy_call_arguments))))
                         for states, actions, episode_discount_weights, episode_policy_call_arguments in
                         zip(self._episodes_states, self._episodes_actions, self._episode_discount_weights,
                             policy_call_arguments)]
        features = tf.pack([tf.concat(0, [log_gradient.flattened_gradient, [1]]) for log_gradient in log_gradients])
        # features = tf.Print(features, [features], message='features', summarize=100)
        # features = tf.Print(features, [tf.shape(features)], message='features shape', summarize=100)
        q_weights = tf.squeeze(util.tensor.least_squares(features, self._episodes_Rs, method='matrix_solve',
                                                         regularize=l2_regularization))
        q_weights = tf.Print(q_weights, [q_weights], message='q_weights', summarize=100)
        q_weights = tf.Print(q_weights, [tf.shape(q_weights)], message='q_weights shape', summarize=100)
        q_weights = tf.Print(q_weights, [tf.matmul(features, tf.reshape(q_weights, [-1, 1]))], message='R', summarize=100)

        policy_gradient = log_gradients[0].new(-q_weights[:log_gradients[0].flattened_gradient.get_shape()[0].value])
        policy_gradient = policy_gradient.reshaped_gradient
        self._update_op = optimizer.apply_gradients(policy_gradient)
        #self._update_op = util.debug.print_gradient(self._update_op, policy_gradient)

    def update(self, state, action, next_state, reward, is_terminal, feed_dict={}):
        self._episode_t += 1

        self._states[-1].append(state)
        self._actions[-1].append(action)
        self._Rs[-1] += self._discount_factor ** self._episode_t * reward
        super().update(state, action, next_state, reward, is_terminal, feed_dict)

    def new_episode(self, feed_dict):
        self._states.append([])
        self._actions.append([])
        self._Rs.append(0)
        self._episode_t = 0

    def episodic_update(self, feed_dict):
        feed_dict[self._episodes_Rs] = self._Rs[:-1]
        print(self._Rs[:-1])
        for episode_states_tensor, episode_states in zip(self._episodes_states, self._states):
            feed_dict[episode_states_tensor] = episode_states
        for episode_actions_tensor, episode_actions in zip(self._episodes_actions, self._actions):
            feed_dict[episode_actions_tensor] = episode_actions
        for discount_weight, states in zip(self._episode_discount_weights, self._states):
            feed_dict[discount_weight] = [self._discount_factor ** t for t in range(len(states))]
        self._update_op.run(feed_dict=feed_dict)

        self._states = [[]]
        self._actions = [[]]
        self._Rs = [0]

    def get_action(self, state):
        return self._policy.sample(state)

if __name__ == '__main__':
    pass
