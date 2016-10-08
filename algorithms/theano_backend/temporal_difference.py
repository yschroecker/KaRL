import theano
import theano.tensor as T
import numpy as np


class TemporalDifferenceLearner:
    def __init__(self, optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, td_error):
        self._optimizer = optimizer

        if loss_clip_threshold is None:
            self._td_loss = T.mean(td_error ** 2)
        elif loss_clip_mode == 'linear':
            td_error = abs(td_error)
            self._td_loss = T.mean(T.minimum(td_error, loss_clip_threshold) ** 2 +
                                   T.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self._td_loss = T.mean(T.clip(td_error, 0, loss_clip_threshold) ** 2)
        else:
            assert False

        self.td_gradient = T.grad(self._td_loss, self._online_weights)
        self._gradient_updates = self._optimizer(self.td_gradient, self._online_weights)
        self._copy_weights = theano.function([], updates=list(zip(self._target_weights, self._online_weights)))


class TemporalDifferenceLearnerQ(TemporalDifferenceLearner):
    def __init__(self, network_builder, optimizer, state_dim, num_actions, discount_factor, td_rule, loss_clip_threshold,
                 loss_clip_mode, create_summaries):
        self.state = T.fmatrix("state")
        self.next_state = T.fmatrix("next_state")
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
            self._max_next_q = self._next_q_target[:, T.argmax(self._next_q_online, axis=1)]
        elif td_rule == 'q-learning':
            self._max_next_q = T.max(self._next_q_target, axis=1)
        elif td_rule == 'gBRM':
            self._next_q_online = self._online_network(self.next_state)
            self._max_next_q = T.max(self._next_q_online, axis=1)
        else:
            assert False, "invalid td_rule for TD-Q"

        max_action = T.argmax(self._q_target, axis=1)
        self._max_action = theano.function([self.state], max_action, allow_input_downcast=True)

        q_a = self._q_online[T.arange(T.shape(self._q_online)[0]), self.action]
        self._action = theano.printing.Print("q_a")(q_a)

        td_error = self.reward + discount_factor * self.target_q_factor * self._max_next_q - q_a

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, td_error)
        self._update = theano.function([self.state, self.action, self.reward, self.next_state, self.target_q_factor],
                                       [q_a, td_error], updates=self._gradient_updates, allow_input_downcast=True)
        self.fixpoint_update()

    def max_action(self, states):
        return self._max_action(states)

    def _update_feed_dict(self, states, actions, rewards, next_states, target_factors):
        feed_dict = {self.state.name: states, self.action.name: actions.astype(np.int32),
                     self.next_state.name: next_states, self.reward.name: rewards,
                     self.target_q_factor.name: target_factors}
        self._feed_dict = feed_dict

    def bellman_operator_update(self, states, actions, rewards, next_states, target_factors):
        self._update_feed_dict(states, actions, rewards, next_states, target_factors)
        return self._update(**self._feed_dict)

    def fixpoint_update(self):
        self._copy_weights()

    def add_summaries(self, summary_writer, episode):
        assert False, "Not implemented"
