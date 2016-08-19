import tensorflow as tf


def index(tensor, index_tensor):
    one_hot_indices = tf.reshape(tf.one_hot(index_tensor, tf.shape(tensor)[1], 1., 0., axis=-1),
                                 [-1, tf.shape(tensor)[1], 1])
    return tf.squeeze(tf.batch_matmul(tf.reshape(tensor, [-1, 1, tf.shape(tensor)[1]]), one_hot_indices))


def nonzero_gradients(gradients):
    return [(tf.maximum(grad, 1e-10), var) for grad, var in gradients]


def mean_gradients(gradients):
    return [(sum([gradient[var_i][0] for gradient in gradients]) / num_sample_episodes, gradients[0][var_i][1])
            for var_i in range(len(gradients[0]))]


def gradient_op(grad1, grad2, op):
    return [(op(grad_element1, grad_element2), var1)
            for (grad_element1, var1), (grad_element2, var2) in zip(grad1, grad2)]


class GradientWrapper:
    def __init__(self, gradient):
        self._gradient = gradient

    def get(self):
        return self._gradient

    def nonzero(self):
        return GradientWrapper(nonzero_gradients(self._gradient))

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)

    def __add__(self, other):
        if type(other) is GradientWrapper:
            return GradientWrapper(gradient_op(self._gradient, other._gradient, lambda x, y: x + y))
        else:
            return GradientWrapper([(grad + other, var) for grad, var in self._gradient])

    def __rsub__(self, other):
        return other.__sub__(self)

    def __sub__(self, other):
        if type(other) is GradientWrapper:
            return GradientWrapper(gradient_op(self._gradient, other._gradient, lambda x, y: x - y))
        else:
            return GradientWrapper([(grad - other, var) for grad, var in self._gradient])

    def __mul__(self, other):
        if type(other) is GradientWrapper:
            return GradientWrapper(gradient_op(self._gradient, other._gradient, lambda x, y: x * y))
        else:
            return GradientWrapper([(grad * other, var) for grad, var in self._gradient])

    def __truediv__(self, other):
        if type(other) is GradientWrapper:
            return GradientWrapper(gradient_op(self._gradient, other._gradient, lambda x, y: x / y))
        else:
            return GradientWrapper([(grad / other, var) for grad, var in self._gradient])


def scope_collection(scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    return tf.get_collection(collection, tf.get_variable_scope().name + "/?" + scope)


def copy_parameters(source_scope, target_scope):
    ops = []
    parent_scope = tf.get_variable_scope().name
    parent_scope_name_length = len(parent_scope) + 1
    if parent_scope_name_length == 1:
        parent_scope_name_length = 0
    for variable in scope_collection(source_scope):
        with tf.variable_op_scope([], target_scope, reuse=True):
            ops.append(tf.get_variable(variable.name[parent_scope_name_length + len(source_scope) + 1:].
                                       split(':', 1)[0]).assign(variable))
    return ops

