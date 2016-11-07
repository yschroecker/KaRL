import abc
import algorithms.theano_backend.policy as policy_backend
import algorithms.theano_backend.policy_gradient as pg_backend
import numpy as np


class EpisodicPG(metaclass=abc.ABCMeta):
    def __init__(self, num_sample_episodes):
        self._num_sample_episodes = num_sample_episodes

        self._new_episode()

    @abc.abstractmethod
    def apply_policy_gradient(self, states, actions, next_states, rewards, is_terminal):
        pass

    def _new_episode(self):
        self._episode_counter = 0
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._is_terminal = []

    def update(self, state, action, next_state, reward, is_terminal):
        self._states.append(state)
        self._actions.append(action)
        self._next_states.append(next_state)
        self._rewards.append(reward)
        self._is_terminal.append(is_terminal)

        if is_terminal:
            self._episode_counter += 1
            if self._episode_counter >= self._num_sample_episodes:
                self.apply_policy_gradient(self._states, self._actions, self._next_states, self._rewards,
                                           self._is_terminal)
                self._new_episode()


class REINFORCE(EpisodicPG):
    def __init__(self, policy, num_sample_episodes, optimizer):
        advantage_estimator = pg_backend.MonteCarloAdvantageEstimator()
        self._policy_gradient_estimator = pg_backend.LikelihoodRatioGradient(
            policy, advantage_estimator, optimizer)
        self._policy = policy_backend.DiscretePolicy(policy)
        super().__init__(num_sample_episodes)

    def apply_policy_gradient(self, states, actions, next_states, rewards, is_terminal):
        self._policy_gradient_estimator.update(states, actions, next_states, rewards,
                                               np.array(is_terminal, dtype=np.float32))

    def get_action(self, state):
        return self._policy.sample(state)


class SimpleActorCritic:
    def __init__(self, policy, critic_learner, discount_factor, actor_optimizer, steps_before_update):
        self._critic = critic_learner
        self._advantage_estimator = pg_backend.TDVAdvantageEstimator(critic_learner, discount_factor)
        self._actor = pg_backend.LikelihoodRatioGradient(policy, self._advantage_estimator,
                                                         actor_optimizer)
        self._policy = policy_backend.DiscretePolicy(policy)

        self._reset_train_set()
        self._steps_before_update = steps_before_update

    def _reset_train_set(self):
        self._step_counter = 0
        self._update_states = []
        self._update_actions = []
        self._update_next_states = []
        self._update_rewards = []
        self._update_is_terminal = []

    def update(self, state, action, next_state, reward, is_terminal):
        self._update_states.append(state)
        self._update_actions.append(action)
        self._update_next_states.append(next_state)
        self._update_rewards.append(reward)
        self._update_is_terminal.append(0. if is_terminal else 1.)

        self._step_counter += 1
        if self._step_counter >= self._steps_before_update:
            update_states = np.array(self._update_states, dtype=np.float32)
            update_actions = np.array(self._update_actions, dtype=np.float32)
            update_next_states = np.array(self._update_next_states, dtype=np.float32)
            update_rewards = np.array(self._update_rewards, dtype=np.float32)
            update_is_terminal = np.array(self._update_is_terminal, dtype=np.float32)

            self._critic.bellman_operator_update(update_states, update_next_states, update_rewards, update_is_terminal)
            self._critic.fixpoint_update()
            self._actor.update(update_states, update_actions, update_next_states, update_rewards, update_is_terminal)
            self._reset_train_set()

    def get_action(self, state):
        return self._policy.sample(state)

