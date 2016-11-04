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
