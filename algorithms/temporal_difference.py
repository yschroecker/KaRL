import numpy as np
import tensorflow as tf
import util.tensor
import util.debug


class TemporalDifferenceLearner:
    def __init__(self, optimizer, loss_clip_threshold,
                 loss_clip_mode, create_summaries, global_step, td_error):
        self._global_step = global_step
        self._optimizer = optimizer

        if loss_clip_threshold is None:
            self.td_loss = tf.reduce_sum(td_error ** 2)
        elif loss_clip_mode == 'linear':
            self.td_loss = tf.reduce_sum(tf.minimum(td_error, loss_clip_threshold) ** 2 +
                                         tf.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self.td_loss = tf.reduce_sum(tf.clip_by_value(td_error, 0, loss_clip_threshold) ** 2)

        self.td_loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'online_network'))
        self.td_loss = tf.Print(self.td_loss, [self.td_loss], message="loss")

        td_gradient = self._optimizer.compute_gradients(self.td_loss, var_list=self._scope_collection('online_network'))

        self.update_op = self._optimizer.apply_gradients(td_gradient, global_step=self._global_step)
        self.update_op = util.debug.print_gradient(self.update_op, td_gradient)
        self.copy_weights_ops = self._copy_weights()

        if create_summaries:
            for variable in self._scope_collection('online_network'):
                tf.histogram_summary(variable.name, variable)
            for variable in self._scope_collection('target_network'):
                tf.histogram_summary(variable.name, variable)
            for gradient, variable in td_gradient:
                tf.histogram_summary(variable.name + "_gradient", gradient)
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


class TemporalDifferenceLearnerQ(TemporalDifferenceLearner):
    def __init__(self, network_builder, optimizer, num_actions, discount_factor, td_rule, loss_clip_threshold,
                 loss_clip_mode, create_summaries, global_step, state, action, reward, next_state, target_factor):
        with tf.variable_op_scope([], 'online_network'):
            self.q = network_builder(state, False)

        with tf.variable_op_scope([], 'target_network'):
            next_q_values_target_net = network_builder(next_state, False)

        if td_rule == 'double-q-learning':
            with tf.variable_op_scope([], 'online_network', reuse=True):
                next_q_values_q_net = network_builder(next_state, True)
            self._max_next_q = util.tensor.index(next_q_values_target_net, tf.argmax(next_q_values_q_net, 1))
        elif td_rule == 'q-learning':
            self._max_next_q = tf.reduce_max(next_q_values_target_net, 1)
        else:
            assert False

        with tf.variable_op_scope([], 'target_network', reuse=True):
            q_values = tf.squeeze(network_builder(state, True))
            self.max_action = tf.squeeze(tf.gather(np.arange(num_actions), tf.argmax(q_values, 0)))

        q_a = util.tensor.index(self.q, action)

        if create_summaries:
            tf.scalar_summary("q", tf.reduce_mean(self.q))
            tf.scalar_summary("sample-R", tf.reduce_mean(reward))

        td_error = reward + discount_factor * target_factor * self._max_next_q - q_a

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, global_step, td_error)


class TemporalDifferenceLearnerV(TemporalDifferenceLearner):
    def __init__(self, network_builder, optimizer, discount_factor, loss_clip_threshold,
                 loss_clip_mode, create_summaries, global_step, state, reward, next_state, target_factor):
        with tf.variable_op_scope([], 'online_network'):
            self.v = network_builder(state, False)

        with tf.variable_op_scope([], 'target_network'):
            self.next_v = network_builder(next_state, False)

        if create_summaries:
            tf.scalar_summary("v", tf.reduce_mean(self.v))
            tf.scalar_summary("sample-R", tf.reduce_mean(reward))

        td_error = reward + discount_factor * target_factor * self.next_v - self.v
        td_error = tf.Print(td_error, [td_error], message='err')
        td_error = tf.Print(td_error, [self.v], message='v')
        td_error = tf.Print(td_error, [self.next_v], message='v')

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, global_step, td_error)
        self.update_op = util.debug.print_gradient(self.update_op, optimizer.compute_gradients(self.v), message='dv')

