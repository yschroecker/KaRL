import numpy as np
import tensorflow as tf
import util.tensor


class TemporalDifferenceLearner:
    def __init__(self, network_builder, optimizer, num_actions, discount_factor, td_rule, loss_clip_threshold,
                 loss_clip_mode, create_summaries, global_step,
                 state, action, reward, next_state, target_q_factor):
        self._global_step = global_step
        build_network = network_builder
        self._optimizer = optimizer

        with tf.variable_op_scope([], 'online_network'):
            self.q = build_network(state, False)

        with tf.variable_op_scope([], 'target_network'):
            next_q_values_target_net = build_network(next_state, False)

        if td_rule == 'double-q-learning':
            with tf.variable_op_scope([], 'online_network', reuse=True):
                next_q_values_q_net = build_network(next_state, True)
            self._max_next_q = util.tensor.index(next_q_values_target_net, tf.argmax(next_q_values_q_net, 1))
        elif td_rule == 'q-learning':
            self._max_next_q = tf.reduce_max(next_q_values_target_net, 1)
        else:
            assert False

        with tf.variable_op_scope([], 'target_network', reuse=True):
            q_values = tf.squeeze(build_network(state, True))
            self.max_action = tf.squeeze(tf.gather(np.arange(num_actions), tf.argmax(q_values, 0)))

        q_a = util.tensor.index(self.q, action)

        td_error = 2 * (reward + discount_factor * target_q_factor * self._max_next_q - q_a)
        if loss_clip_threshold is None:
            self.td_loss = tf.reduce_sum(td_error ** 2)
        elif loss_clip_mode == 'linear':
            self.td_loss = tf.reduce_sum(tf.minimum(td_error, loss_clip_threshold) ** 2 +
                                         tf.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self.td_loss = tf.reduce_sum(tf.clip_by_value(td_error, 0, loss_clip_threshold) ** 2)

        self.td_loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'online_network'))

        td_gradient = self._optimizer.compute_gradients(self.td_loss, self._scope_collection('online_network'))

        self.update_op = self._optimizer.apply_gradients(td_gradient, global_step=self._global_step)
        self.copy_weights_ops = self._copy_weights()

        if create_summaries:
            for variable in self._scope_collection('online_network'):
                tf.histogram_summary(variable.name, variable)
            for variable in self._scope_collection('target_network'):
                tf.histogram_summary(variable.name, variable)
            for gradient, variable in td_gradient:
                tf.histogram_summary(variable.name + "_gradient", gradient)
            tf.scalar_summary("sample-R", tf.reduce_mean(reward))
            tf.scalar_summary("q", tf.reduce_mean(self.q))
            tf.scalar_summary("td loss", self.td_loss)
            self.summary_op = tf.merge_all_summaries()

    def _copy_weights(self):
        ops = []
        with tf.variable_op_scope([], 'target_network', reuse=True):
            for variable in self._scope_collection('online_network'):
                ops.append(tf.get_variable(variable.name.split('/', 1)[1].split(':', 1)[0]).assign(variable))
        return ops

    @staticmethod
    def _scope_collection(scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
