import tensorflow as tf


def print_gradient(update_op, gradient, message=""):
    ops = []
    for grad, var in gradient:
        if grad is not None:
            ops.append(tf.Print(grad, [grad], summarize=1000, message=message))
    with tf.control_dependencies(ops + [update_op]):
        return tf.no_op()

