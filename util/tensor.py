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


class GradientVector:
    _VariableDescriptor = collections.namedtuple('_VariableDescriptor', ['range', 'var'])

    def __init__(self, gradient):
        self._gradient = gradient

        flattened_gradient = []
        self._descriptors = []
        current_index = 0
        for grad, var in gradient:
            dim = np.prod(var.get_shape()).value
            flattened_gradient.append(tf.reshape(grad, shape=[-1]))
            self._descriptors.append(self._VariableDescriptor((current_index, current_index + dim), var))
            current_index += dim

        self.flattened_gradient = tf.concat(0, flattened_gradient)

    @property
    def reshaped_gradient(self):
        gradient = []
        for variable_descriptor in self._descriptors:
            variable_gradient = self.flattened_gradient[variable_descriptor.range[0]:variable_descriptor.range[1]]
            variable_gradient = tf.reshape(variable_gradient, shape=variable_descriptor.var.get_shape())
            gradient.append((variable_gradient, variable_descriptor.var))
        return gradient

    def new(self, flat_gradient):
        new_gradient = type(self).__new__(type(self))
        new_gradient.flattened_gradient = flat_gradient
        new_gradient._descriptors = self._descriptors
        return new_gradient


def scope_collection(scope='', collection=tf.GraphKeys.TRAINABLE_VARIABLES):
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


def least_squares(X, y, method='svd', regularize=0):
    if method == 'matrix_solve':
        XTX = tf.matmul(tf.transpose(X), X) + tf.diag(regularize * tf.ones([tf.shape(X)[1]]))
        XTy = tf.matmul(tf.transpose(X), tf.reshape(y, shape=[-1, 1]))
        return tf.matrix_solve(XTX, XTy)
    elif method == 'svd':
        assert regularize == 0
        s, u, v = tf.svd(X)
        s_inv = 1/s
        return tf.matmul(tf.matmul(tf.matmul(v, tf.diag(s_inv), transpose_a=True), u, transpose_b=True),
                         tf.reshape(y, shape=[-1, 1]))
    else:
        assert False


def var_name(var):
    return var.name.split('/')[-1].split(':')[0]


def test_parameter_vector_descriptor():
    from examples.lander.networks import build_policy_network
    import debug
    sess = tf.InteractiveSession()
    state = tf.placeholder(tf.float32, shape=[None, 4])
    network = build_policy_network(state)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    gradient = optimizer.compute_gradients(tf.log(network))
    wrapper = GradientVector(gradient)
    wrapper.flattened_gradient *= 2
    op = debug.print_gradient(tf.no_op(), gradient)
    op = debug.print_gradient(op, wrapper.reshaped_gradient)
    tf.initialize_all_variables().run()
    op.run(feed_dict={state: [[4, 3, 1, 2]]})



def test_least_squares():
    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None])
    tf.initialize_all_variables().run()
    for method in ['matrix_solve', 'svd']:
        p = least_squares(X, y, method=method)
        print(p.eval(feed_dict={X: [[1], [3], [5]], y: [4.02, 11.95, 19.98]}))

if __name__ == '__main__':
    test_least_squares()
    # test_parameter_vector_descriptor()




