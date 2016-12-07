import theano
import theano.gradient
import theano.tensor as T
import numpy as np
import abc
import algorithms.theano_backend.bokehboard


class TemporalDifferenceLearner(metaclass=abc.ABCMeta):
    def __init__(self, optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, td_error):
        self._optimizer = optimizer

        if loss_clip_threshold is None:
            self._td_loss = T.sum(td_error ** 2)
        elif loss_clip_mode == 'linear':
            td_error = abs(td_error)
            self._td_loss = T.sum(T.minimum(td_error, loss_clip_threshold) ** 2 +
                                  T.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self._td_loss = T.sum(T.clip(td_error, 0, loss_clip_threshold) ** 2)
        else:
            assert False

        self.td_gradient = T.grad(self._td_loss, self._online_weights)
        self._gradient_updates = self._optimizer(self.td_gradient, self._online_weights)
        self._copy_weights = theano.function([], updates=list(zip(self._target_weights, self._online_weights)))
        self._create_summaries = create_summaries
        if create_summaries:
            self._bokehboard = algorithms.theano_backend.bokehboard.Bokehboard()
            network = self._bokehboard.board_builder.add_network("Online TD Network")
            for online_weight in self._online_weights:
                network.add_parameter("online." + online_weight.name, online_weight, self._gradient_name(online_weight))

        self._dbg_count = 0

    def fixpoint_update(self):
        self._copy_weights()

    def bellman_operator_update(self, *args, **kwargs):
        if self._create_summaries and self._bokehboard.ready_for_update():
            gradients = self._update_with_get(*args, **kwargs)
            for var, var_gradient in zip(self._online_weights, gradients):
                name = self._gradient_name(var)
                self._bokehboard.update_python_variable(**{name: var_gradient})
        else:
            self._update(*args, **kwargs)

    def _gradient_name(self, online_weight):
        return "online.{}.gradient".format(online_weight.name)


class TemporalDifferenceLearnerQ(TemporalDifferenceLearner):
    def __init__(self, network_builder, optimizer, state_dim, num_actions, discount_factor, td_rule,
                 loss_clip_threshold, loss_clip_mode, create_summaries):
        state_tensor_type = T.TensorType('float32', (False,)*(len(state_dim) + 1))
        self.state = state_tensor_type("state")
        self.next_state = state_tensor_type("next_state")
        self.action = T.ivector("action")
        self.reward = T.fvector("reward")
        self.target_q_factor = T.fvector("target_q_factor")  # 0 for terminal states.

        self._online_network, self._online_weights = network_builder()
        self._target_network, self._target_weights = network_builder()
        self._q_online = self._online_network(self.state)
        self._q_target = self._target_network(self.state)
        self._next_q_target = self._target_network(self.next_state)

        if td_rule == 'double-q-learning':
            # with tf.variable_op_scope([], 'online_network', reuse=True):
            #     next_q_values_q_net = network_builder(self.next_state, True)
            # self._max_next_q = util.tensor.index(next_q_values_target_net, tf.argmax(next_q_values_q_net, 1))
            self._next_q_online = self._online_network(self.next_state)
            self._max_next_q = self._next_q_target[T.arange(T.shape(self._next_q_target)[0]),
                                                   T.argmax(self._next_q_target, axis=1)]
        elif td_rule == 'q-learning':
            self._max_next_q = T.max(self._next_q_target, axis=1)
        elif td_rule == 'gBRM':
            self._next_q_online = self._online_network(self.next_state)
            self._max_next_q = T.max(self._next_q_online, axis=1)
        else:
            assert False, "invalid td_rule for TD-Q"
        self._max_next_q = theano.gradient.disconnected_grad(self._max_next_q)

        max_action = T.argmax(self._q_target, axis=1)
        self._max_action = theano.function([self.state], max_action, allow_input_downcast=True)

        q_a = self._q_online[T.arange(T.shape(self._q_online)[0]), self.action]

        td_error = self.reward + discount_factor * self.target_q_factor * self._max_next_q - q_a

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, td_error)
        mean_q = T.mean(q_a)
        mean_r = T.mean(self.reward)
        mean_td_error = T.mean(td_error)
        self._update_fn = theano.function([self.state, self.action, self.reward, self.next_state, self.target_q_factor],
                                          [mean_q, mean_r, mean_td_error], updates=self._gradient_updates,
                                          allow_input_downcast=True)
        self._update_with_get_fn = theano.function([self.state, self.action, self.reward, self.next_state,
                                                    self.target_q_factor], [mean_q, mean_r, mean_td_error] +
                                                   self.td_gradient,
                                                   updates=self._gradient_updates, allow_input_downcast=True)
        self.fixpoint_update()

        if create_summaries:
            self._bokehboard.board_builder.add_statistic_pyvar("sample_q", 0)
            self._bokehboard.board_builder.add_statistic_pyvar("sample_r", 0)
            self._bokehboard.board_builder.add_statistic_pyvar("td_error", 0)

    def _update(self, *args, **kwargs):
        q_a, reward, td_error = self._update_fn(*args, **kwargs)
        self._add_summary_statistics(q_a, reward, td_error)

    def _update_with_get(self, *args, **kwargs):
        result = self._update_with_get_fn(*args, **kwargs)
        q_a, reward, td_error = result[0:3]
        self._add_summary_statistics(q_a, reward, td_error)
        return result[3:]

    def _add_summary_statistics(self, q_a, reward, td_error):
        if self._create_summaries:
            self._bokehboard.update_python_variable(sample_q=q_a, sample_r=reward, td_error=td_error)

    def max_action(self, states):
        return self._max_action(states)

    def _update_feed_dict(self, states, actions, rewards, next_states, target_factors):
        feed_dict = {self.state.name: states, self.action.name: actions.astype(np.int32),
                     self.next_state.name: next_states, self.reward.name: rewards,
                     self.target_q_factor.name: target_factors}
        self._feed_dict = feed_dict

    def bellman_operator_update(self, states, actions, next_states, rewards, target_factors):
        self._update_feed_dict(states, actions, rewards, next_states, target_factors)
        return super().bellman_operator_update(**self._feed_dict)

    def add_summaries(self, summary_writer, episode):
        raise NotImplementedError()

    def state_action_value(self, states, actions):
        return self._online_network(states, actions)


class TemporalDifferenceLearnerV(TemporalDifferenceLearner):
    def __init__(self, network_builder, optimizer, state_dim, discount_factor, td_rule,
                 loss_clip_threshold, loss_clip_mode, create_summaries):
        self._online_network, self._online_weights = network_builder()
        self._target_network, self._target_weights = network_builder()

        state_tensor_type = T.TensorType('float32', (False,)*(len(state_dim) + 1))
        self.state = state_tensor_type("state")
        self.next_state = state_tensor_type("next_state")
        self.reward = T.fvector("reward")
        self.target_factor = T.fvector("target_factor")

        self._v_online = self._online_network(self.state)[:, 0]
        self._next_v_target = self._target_network(self.next_state)[:, 0]

        if td_rule == '1-step':
            td_error = self.reward + discount_factor * self.target_factor * self._next_v_target - self._v_online
        else:
            assert False

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, td_error)
        self._update = theano.function([self.state, self.next_state, self.reward, self.target_factor],
                                       updates=self._gradient_updates, allow_input_downcast=True)
        self._update_with_get = theano.function([self.state, self.next_state, self.reward, self.target_factor],
                                                self.td_gradient, updates=self._gradient_updates,
                                                allow_input_downcast=True)
        self.fixpoint_update()

    def value(self, states):
        return self._online_network(states)[:, 0]

