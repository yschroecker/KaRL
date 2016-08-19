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
            self.td_loss = tf.reduce_mean(td_error ** 2)
        elif loss_clip_mode == 'linear':
            self.td_loss = tf.reduce_mean(tf.minimum(td_error, loss_clip_threshold) ** 2 +
                                          tf.maximum(td_error - loss_clip_threshold, 0))
        elif loss_clip_mode == 'absolute':
            self.td_loss = tf.reduce_mean(tf.clip_by_value(td_error, 0, loss_clip_threshold) ** 2)

        self.td_loss += sum(util.tensor.scope_collection('online_network', tf.GraphKeys.REGULARIZATION_LOSSES))
        # self.td_loss = tf.Print(self.td_loss, [self.td_loss], message="loss")

        self.td_gradient = self._optimizer.compute_gradients(self.td_loss,
                                                             var_list=util.tensor.scope_collection('online_network'))

        # self.update_op = self._optimizer.apply_gradients(td_gradient, global_step=self._global_step)
        # self.update_op = util.debug.print_gradient(self.update_op, td_gradient, message='dtd')
        self.copy_weights_ops = util.tensor.copy_parameters('online_network', 'target_network')

        if create_summaries:
            for variable in util.tensor.scope_collection('online_network'):
                tf.histogram_summary(variable.name, variable)
            for variable in util.tensor.scope_collection('target_network'):
                tf.histogram_summary(variable.name, variable)
            for gradient, variable in self.td_gradient:
                tf.histogram_summary(variable.name + "_gradient", gradient)
            tf.scalar_summary("td loss", self.td_loss)
            self.summary_op = tf.merge_all_summaries()


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
            assert False, "invalid td_rule for TD-Q"

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
    def __init__(self, network_builder, optimizer, discount_factor, td_rule, loss_clip_threshold,
                 loss_clip_mode, create_summaries, global_step, state, reward, next_state, target_factor):
        with tf.variable_op_scope([], 'online_network'):
            self.v = tf.squeeze(network_builder(state, False), squeeze_dims=[1])

        with tf.variable_op_scope([], 'target_network'):
            self.next_v = tf.squeeze(network_builder(next_state, False), squeeze_dims=[1])

        if create_summaries:
            tf.scalar_summary("v", tf.reduce_mean(self.v))
            tf.scalar_summary("sample-R", tf.reduce_mean(reward))

        if td_rule == '1-step':
            td_error = reward + discount_factor * target_factor * self.next_v - self.v
        elif td_rule == 'deepmind-n-step':
            last_v = tf.gather(self.next_v, tf.size(self.next_v) - 1)
            target = tf.reverse(
                tf.scan(fn=lambda R, r: r + discount_factor * R,
                        elems=tf.reverse(reward, dims=[True]),
                        initializer=last_v), dims=[True])

            td_error = target - self.v
        else:
            assert False, "invalid td_rule for TD-V"
        # td_error = tf.Print(td_error, [reward], message='R', summarize=1000)
        # # td_error = tf.Print(td_error, [tf.shape(reward)], message="R.shape", summarize=1000)
        # # td_error = tf.Print(td_error, [tf.shape(td_error)], message='err.shape', summarize=1000)
        # # td_error = tf.Print(td_error, [td_error], message='err', summarize=1000)
        # td_error = tf.Print(td_error, [self.v], message='v', summarize=1000)
        # # td_error = tf.Print(td_error, [tf.shape(self.v)], message='v.shape', summarize=1000)
        # td_error = tf.Print(td_error, [self.next_v], message="v'", summarize
        # target = tf.Print(target, [target], message="target", summarize=100)   # td_error = tf.Print(td_error, [tf.shape(self.next_v)], message="v'.shape", summarize=1000)
        # td_error = tf.Print(td_error, [target_factor], message='f', summarize=1000)

        super().__init__(optimizer, loss_clip_threshold, loss_clip_mode, create_summaries, global_step, td_error)
        #self.update_op = util.debug.print_gradient(self.update_op, optimizer.compute_gradients(self.v), message='dv')

