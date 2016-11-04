import gridworld
import numpy as np
import tensorflow as tf

import algorithms.async as async
import algorithms.tensorflow_backend.policy_gradient as pg


def build_policy_network(states):
    states = tf.cast(states, tf.int64)
    state = tf.one_hot(states, gridworld.width * gridworld.height, 1, 0, axis=-1)
    input_dim = gridworld.width * gridworld.height
    state = tf.reshape(tf.cast(state, tf.float32), [-1, input_dim])

    weights = tf.get_variable("policy_weights", shape=[input_dim, gridworld.num_actions],
                              initializer=tf.truncated_normal_initializer())
    # weights = tf.Print(weights, [weights], message=tf.get_variable_scope().name, summarize=100)

    return tf.nn.softmax(tf.matmul(state, weights))


def build_value_network(states, reuse):
    states = tf.cast(states, tf.int64)
    state = tf.one_hot(states, gridworld.width * gridworld.height, 1, 0, axis=-1)
    input_dim = gridworld.width * gridworld.height
    state = tf.reshape(tf.cast(state, tf.float32), [-1, input_dim])

    weights = tf.get_variable("value_weights", shape=[input_dim, 1],
                              initializer=tf.constant_initializer(1))

    return tf.matmul(state, weights)

num_instances = 10


def build_algorithm():
    actor_optimizer = tf.train.RMSPropOptimizer(0.5 / num_instances)
    critic_optimizer = tf.train.GradientDescentOptimizer(0.5 / num_instances)
    policy = pg.DiscretePolicy([1], 4, build_policy_network, actor_optimizer)
    return async.A3C(policy=policy, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                     state_dim=[1], discount_factor=gridworld.discount_factor,
                     value_network_builder=build_value_network)

if __name__ == '__main__':
    np.seterr(all='raise')
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            learner = async.Async(algorithm=build_algorithm, num_instances=num_instances)
            session.run(tf.initialize_all_variables())

            current_state = [(0, 0) for _ in range(num_instances)]
            cumulative_reward = [0 for _ in range(num_instances)]
            episode_t = [0 for _ in range(num_instances)]
            for t in range(500000):
                for instance in range(num_instances):
                    action_ = learner.get_action(instance, gridworld.state_index(current_state[instance]))
                    next_state_ = gridworld.transition(current_state[instance], action_)
                    reward_ = gridworld.reward_function(next_state_)
                    cumulative_reward[instance] += gridworld.discount_factor ** episode_t[instance] * reward_
                    episode_t[instance] += 1
                    is_terminal = gridworld.is_terminal(next_state_)
                    learner.update(instance, gridworld.state_index(current_state[instance]), action_,
                                   gridworld.state_index(next_state_), reward_, is_terminal)
                    if is_terminal:
                        current_state[instance] = (0, 0)

                        print("Finished episode in t=%d, instance=%d - reward:%f" %
                              (t, instance, cumulative_reward[instance]))
                        cumulative_reward[instance] = 0
                        episode_t[instance] = 0
                    else:
                        current_state[instance] = next_state_
