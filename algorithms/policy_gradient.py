import tensorflow as tf
import numpy as np
import util.tensor
import util.debug
import algorithms.temporal_difference as td


class DiscretePolicy:
    def __init__(self, state_dim, num_actions, policy_network_builder, optimizer):
        self._build_policy_network = policy_network_builder
        self._num_actions = num_actions
        self._states = tf.placeholder(tf.float32, shape=[None] + state_dim, name="states")

        with tf.variable_scope('policy_network'):
            self._policy = policy_network_builder(self._states)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy_network')

    def log_policy(self, state, action):
        with tf.variable_scope('policy_network', reuse=True):
            policy = self._build_policy_network(state)
            policy_action = util.tensor.index(policy, action)
        return tf.log(policy_action)

    def sample(self, state):
        probabilities = self._policy.eval(feed_dict={self._states: [state]})[0]
        return np.random.choice(np.arange(self._num_actions), p=probabilities)


class AdvantageActorCritic:
    def __init__(self, state_dim, policy, value_network_builder, actor_optimizer, critic_optimizer,
                 discount_factor):

        self._policy = policy

        self._empty_observervations()

        self._state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="state")
        self._action = tf.placeholder(tf.int32, shape=[None], name="action")
        self._next_state = tf.placeholder(tf.float32, shape=[None] + state_dim, name="next_state")
        self._reward = tf.placeholder(tf.float32, shape=[None], name="reward")
        self._target_factor = tf.placeholder(tf.float32, shape=[None])  # 0 for terminal states.

        self._td_learner = td.TemporalDifferenceLearnerV(value_network_builder, critic_optimizer, discount_factor,
                                                         1, 'linear', False,  # TODO: make parameters
                                                         tf.get_variable("ac_step", shape=[], dtype=tf.int32,
                                                                         initializer=tf.constant_initializer(0),
                                                                         trainable=False),
                                                         self._state, self._reward, self._next_state,
                                                         self._target_factor)

        # TODO: check if target or online network is used correctly
        advantages = self._reward + discount_factor * self._td_learner.next_v - self._td_learner.v
        policy_gradient = actor_optimizer.compute_gradients(-policy.log_policy(self._state, self._action) *
                                                            advantages)
        self._actor_update = actor_optimizer.apply_gradients(policy_gradient)

    def update(self, state, action, next_state, reward, is_terminal):
        # TODO: make one op
        self._observed_states.append(state)
        self._observed_actions.append(action)
        self._observed_next_states.append(next_state)
        self._observed_rewards.append(reward)
        self._observed_target_factors.append(0 if is_terminal else 1)

        if is_terminal:  # TODO: generalize
            feed_dict = {self._state: self._observed_states,
                         self._action: self._observed_actions,
                         self._next_state: self._observed_next_states,
                         self._reward: self._observed_rewards,
                         self._target_factor: self._observed_target_factors}

            self._actor_update.run(feed_dict=feed_dict)
            self._td_learner.update_op.run(feed_dict=feed_dict)
            tf.get_default_session().run(self._td_learner.copy_weights_ops, feed_dict=feed_dict)

            self._empty_observervations()

    def get_action(self, state):
        return self._policy.sample(state)

    def _empty_observervations(self):
        self._observed_states = []
        self._observed_actions = []
        self._observed_next_states = []
        self._observed_rewards = []
        self._observed_target_factors = []


class REINFORCE:
    def __init__(self, state_dim, policy, optimizer, discount_factor, num_sample_episodes=3, baseline='constant'):
        self._states = [[]]
        self._actions = [[]]
        self._qs = [[]]

        self._episodes_states = [tf.placeholder(tf.float32, shape=[None] + state_dim)
                                 for _ in range(num_sample_episodes)]
        self._episodes_actions = [tf.placeholder(tf.int32, shape=[None]) for _ in range(num_sample_episodes)]
        self._episodes_qs = [tf.placeholder(tf.float32, shape=[None]) for _ in range(num_sample_episodes)]
        self._policy = policy
        self._discount_factor = discount_factor
        self._episode_t = 0
        self._episode_counter = 0
        self._num_sample_episodes = num_sample_episodes

        # REINFORCE gradient

        #stepwise REINFORCE gradient:
        # \sum_{i=1}^N \sum_{t=1}^{T} (\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \sum_{h=t}^T r_h )
        # grad = E[SUM_t(dlogp_t Q)]
        #Optimal baseline
        # b = E[SUM_t(dlogp_t) SUM_t(dlogp_t Q)]/E[SUM_t(dlogp_t)²]
        # grad = E[SUM_t(dlogp_t (Q - b))]
        log_policies = [policy.log_policy(states, actions)
                        for states, actions in zip(self._episodes_states, self._episodes_actions)]
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

        policy_gradient = (sum(e_p_g_with_baseline)/num_sample_episodes).get()

        self._update_op = optimizer.apply_gradients(policy_gradient)

    def update(self, state, action, next_state, reward, is_terminal):
        self._episode_t += 1

        self._states[-1].append(state)
        self._actions[-1].append(action)
        for t in range(self._episode_t - 1):
            self._qs[-1][t] += self._discount_factor ** (self._episode_t - t) * reward
        self._qs[-1].append(reward)
        if is_terminal:
            self.new_episode()

    def new_episode(self):
        self._episode_counter += 1
        if self._episode_counter >= self._num_sample_episodes:
            feed_dict = {}
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
            self._episode_counter = 0
        else:
            self._states.append([])
            self._actions.append([])
            self._qs.append([])
        self._episode_t = 0

    def get_action(self, state):
        return self._policy.sample(state)

if __name__ == '__main__':
    pass
