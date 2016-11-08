import examples.gridworld.gridworld as gridworld
import numpy as np
import theano.tensor as T
import lasagne
import functools

import algorithms.policy_gradient as pg
import algorithms.theano_backend.policy as policy
import algorithms.theano_backend.temporal_difference as td


def build_policy_network():
    # states = tf.cast(states, tf.int64)
    # state = tf.one_hot(states, gridworld.width * gridworld.height, 1, 0, axis=-1)
    # input_dim = gridworld.width * gridworld.height
    # state = tf.reshape(tf.cast(state, tf.float32), [-1, input_dim])
    #
    # weights = tf.get_variable("policy_weights", shape=[input_dim, gridworld.num_actions],
    #                           initializer=tf.truncated_normal_initializer())
    #
    # return tf.nn.softmax(tf.matmul(state, weights))
    l_in = lasagne.layers.InputLayer((None, gridworld.width * gridworld.height))
    l_out = lasagne.layers.DenseLayer(l_in, gridworld.num_actions, nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.HeUniform())
    return lambda s: lasagne.layers.get_output(l_out, T.extra_ops.to_one_hot(T.cast(s, 'int32'),
                                                                             gridworld.width * gridworld.height)), \
           lasagne.layers.get_all_params(l_out)


def build_value_network():
    l_in = lasagne.layers.InputLayer((None, gridworld.width * gridworld.height))
    l_out = lasagne.layers.DenseLayer(l_in, 1, nonlinearity=lasagne.nonlinearities.linear,
                                      W=lasagne.init.HeUniform())
    return lambda s: lasagne.layers.get_output(l_out, T.extra_ops.to_one_hot(T.cast(s, 'int32'),
                                                                             gridworld.width * gridworld.height)), \
           lasagne.layers.get_all_params(l_out)

if __name__ == '__main__':
    np.seterr(all='raise')
    # actor_optimizer = tf.train.RMSPropOptimizer(0.5)
    # critic_optimizer = tf.train.GradientDescentOptimizer(0.5)
    actor_optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=0.005)
    critic_optimizer = functools.partial(lasagne.updates.sgd, learning_rate=0.05)
    # policy = pg.DiscretePolicy([1], 4, build_policy_network, actor_optimizer)
    actor = policy.DiscreteTensorPolicy(gridworld.num_actions, build_policy_network)
    critic = td.TemporalDifferenceLearnerV(build_value_network, critic_optimizer, [gridworld.width * gridworld.height],
                                           gridworld.discount_factor, '1-step', None, None, False)
    learner = pg.SimpleActorCritic(actor, critic, gridworld.discount_factor, actor_optimizer, 1)
    # learner = pg.S(policy=policy, value_network_builder=build_value_network, state_dim=[1],
    #                                   discount_factor=gridworld.discount_factor,
    #                                   actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)

    current_state = (0, 0)
    cumulative_reward = 0
    episode_t = 0
    for t in range(500000):
        action_ = learner.get_action(gridworld.state_index(current_state))
        next_state_ = gridworld.transition(current_state, action_)
        reward_ = gridworld.reward_function(next_state_)
        cumulative_reward += gridworld.discount_factor ** episode_t * reward_
        episode_t += 1
        is_terminal = gridworld.is_terminal(next_state_)
        learner.update(gridworld.state_index(current_state), action_, gridworld.state_index(next_state_),
                       reward_, is_terminal)
        if is_terminal:
            current_state = (0, 0)

            print("Finished episode in t=%d - reward:%f " % (t, cumulative_reward))
            cumulative_reward = 0
            episode_t = 0
        else:
            current_state = next_state_