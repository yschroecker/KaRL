import theano
import theano.tensor as T
import numpy as np


class DiscreteTensorPolicy:
    def __init__(self, num_actions, policy_network_builder):
        self.num_actions = num_actions
        self.action_dtype = 'int32'

        self._policy, self.params = policy_network_builder()

    def __call__(self, state, action=None):
        action_probabilities = self._policy(state)
        if action is None:
            return action_probabilities
        else:
            return action_probabilities[T.arange(T.shape(state)[0]), action]

    def log(self, state, action):
        return T.log(self(state, action))

    def save(self, path):
        raise NotImplementedError()  # TODO

    @classmethod
    def load(cls, path):
        raise NotImplementedError()  # TODO


class DiscretePolicy:
    def __init__(self, tensor_policy):
        self._tensor_policy = tensor_policy

        states_tensor = T.fmatrix('states')
        state_probabilities_tensor = self._tensor_policy(states_tensor)
        self._state_probabilities = theano.function(
            [states_tensor], [state_probabilities_tensor], allow_input_downcast=True)

    def state_probabilities(self, state):
        return self._state_probabilities(np.array([state]))[0]

    def sample(self, state):
        probabilities = self.state_probabilities(state)[0]
        return np.random.choice(np.arange(self._tensor_policy.num_actions), p=probabilities)

    def greedy(self, state):
        probabilities = self.state_probabilities(state)
        return np.argmax(probabilities)

