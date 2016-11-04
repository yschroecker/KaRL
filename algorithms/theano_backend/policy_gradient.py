import theano
import theano.tensor as T
import theano.tensor.extra_ops as extra_ops
import numpy as np


class LikelihoodRatioGradient:
    def __init__(self, policy, advantage_estimator, optimizer):
        self._policy = policy

        states = T.fmatrix("states")
        actions = T.vector('actions', policy.action_dtype)
        next_states = T.fmatrix("next_states")
        rewards = T.fvector("rewards")
        is_terminal = T.fvector("is_terminal")

        advantages = advantage_estimator.advantages(states, actions, next_states, rewards, is_terminal, policy)
        policy_gradient = self._policy_gradient(states, actions, advantages, is_terminal)
        gradient_updates = optimizer(policy_gradient, policy.params)

        self.update = theano.function([states, actions, next_states, rewards, is_terminal], updates=gradient_updates,
                                      on_unused_input='ignore', allow_input_downcast=True)

    def _policy_gradient(self, states, actions, advantages, is_terminal):
        log_action_probabilities = self._policy.log(states, actions)
        return T.grad(1/T.sum(is_terminal) * -T.sum(log_action_probabilities * advantages, axis=0),
                      self._policy.params, disconnected_inputs=advantages)


class MonteCarloAdvantageEstimator:
    def advantages(self, states, actions, next_states, rewards, is_terminal, policy):
        """
        :param states:
            All states of one episode
        :param actions:
            All actions of one episode
        :param rewards:
            All rewards of one epiosde
        :return:
            Advantage values of one episode
        """
        # def mean_over_trajectories(val):
        #
        #     beginnings = T.concatenate((np.array([1], dtype=np.float32), is_terminal[:-1]))
        #     new_indices = theano.scan(lambda beginning, index: (1 - beginning) * index + 1,
        #                               outputs_info=T.as_tensor(np.float32(0)),
        #                               sequences=[beginnings])[0] - 1
        #     new_indices = T.cast(new_indices, 'int32')
        #     max_index = T.max(new_indices)
        #
        #     def sum_over_trajectories(val):
        #         return theano.reduce(lambda index, value, result: T.set_subtensor(result[index:index+1],
        #                                                                           result[index:index+1] + value),
        #                              outputs_info=T.zeros_like(val),
        #                              sequences=[new_indices, val])[0]
        #     mean_values = sum_over_trajectories(val)/sum_over_trajectories(T.ones_like(val))
        #     return mean_values[new_indices]

        state_action_values = theano.scan(lambda reward, is_terminal, value: reward + value * (1 - is_terminal),
                                          outputs_info=T.as_tensor(np.float32(0)),
                                          go_backwards=True,
                                          sequences=[rewards, is_terminal])[0][::-1]
        log_action_probabilities = policy.log(states, actions)

        # log_policy_derivative = T.jacobian(log_action_probabilities, policy.params)
        # baseline = []
        # for i, var_derivative in enumerate(log_policy_derivative):
        #     squared_log_policy_derivative = var_derivative**2
        #     baseline_numerator = state_action_values * squared_log_policy_derivative
        #     baseline_denominator = squared_log_policy_derivative
        #     baseline.append(mean_over_trajectories(baseline_numerator)/mean_over_trajectories(baseline_denominator))
        return state_action_values


if __name__ == '__main__':
    def mean_over_trajectories(val, is_terminal):

        beginnings = T.concatenate((np.array([1], dtype=np.float32), is_terminal[:-1]))
        new_indices = theano.scan(lambda beginning, index: (1 - beginning) * index + 1,
                                  outputs_info=T.as_tensor(np.float32(0)),
                                  sequences=[beginnings])[0] - 1
        new_indices = T.cast(new_indices, 'int32')
        max_index = T.max(new_indices)
        def sum_over_trajectories(val):
            return theano.reduce(lambda index, value, result: T.set_subtensor(result[index:index+1],
                                                                              result[index:index+1] + value),
                                 outputs_info=T.zeros((max_index + 1, )),
                                 sequences=[new_indices, val])[0]
        mean_values = sum_over_trajectories(val)/sum_over_trajectories(T.ones_like(val))
        return mean_values[new_indices]

    mc_advantage_estimator = MonteCarloAdvantageEstimator()
    # states = T.fmatrix("states")
    # actions = T.fvector('actions')
    # next_states = T.fmatrix("next_states")
    rewards = T.fvector("rewards")
    is_terminal = T.fvector("is_terminal")
    # advantages_t = mc_advantage_estimator.advantages(states, actions, next_states, rewards, is_terminal, None)
    # advantages = theano.function([states, actions, next_states, rewards, is_terminal], [advantages_t], on_unused_input='ignore', allow_input_downcast=True)
    # print(advantages(np.zeros((10, 3)), np.zeros((10,)), np.zeros((10, 3)),
    #                  np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32), np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)))

    indices = mean_over_trajectories(rewards, is_terminal)
    indices = theano.function([rewards, is_terminal], [indices], allow_input_downcast=True)
    indices = indices(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32),
                      np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32))
    print(indices)
