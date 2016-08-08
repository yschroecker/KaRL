import gridworld
import tensorflow as tf
import algorithms.policy_gradient as pg


def build_network(states):
    states = tf.cast(states, tf.int64)
    state = tf.one_hot(states, gridworld.width * gridworld.height, 1, 0, axis=-1)
    input_dim = gridworld.width * gridworld.height
    state = tf.reshape(tf.cast(state, tf.float32), [-1, input_dim])

    weights = tf.get_variable("weights", shape=[input_dim, gridworld.num_actions],
                              initializer=tf.truncated_normal_initializer())

    return tf.nn.softmax(tf.matmul(state, weights))


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            optimizer = tf.train.RMSPropOptimizer(0.5)
            policy = pg.DiscretePolicy([1], 4, build_network, optimizer)
            learner = pg.REINFORCE(policy=policy, state_dim=[1], discount_factor=gridworld.discount_factor, optimizer=optimizer)
            session.run(tf.initialize_all_variables())

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

                    print("Finished episode in t=%d - reward:%f" % (t, cumulative_reward))
                    cumulative_reward = 0
                    episode_t = 0
                else:
                    current_state = next_state_
