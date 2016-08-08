import tensorflow as tf


def print_gradient(update_op, gradient):
    ops = []
    for grad, var in gradient:
        ops.append(tf.Print(grad, [grad], summarize=1000))
    with tf.control_dependencies(ops + [update_op]):
        return tf.no_op()

