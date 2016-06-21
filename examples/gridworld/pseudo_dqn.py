import tensorflow as tf
from gridworld import *
import algorithms.dqn as dqn


class PseudoQNetwork:
    def __init__(self):
        self.action_dim = []
        self.discretized_actions = list(range(num_actions))
        self.state_dim = []

    def build_network(self, state, action):
        state = tf.cast(state, tf.int64)
        action = tf.cast(action, tf.int64)
        state_action = tf.concat(0, [tf.one_hot(state, width * height, 1, 0),
                                     tf.one_hot(action, num_actions, 1, 0)])
        input_dim = width * height + num_actions
        state_action = tf.reshape(tf.cast(state_action, tf.float32), [1, input_dim])

        weights = tf.get_variable("weights", shape=[input_dim, 1], initializer=tf.constant_initializer(1))
        return tf.matmul(state_action, weights)


def state_index(state):
    return state[0] * width + state[1]


if __name__ == '__main__':
    with tf.Session() as session:
        q_network = PseudoQNetwork()
        learner = dqn.DQN(q_network, 1, discount_factor)
        eps_greedy = dqn.EpsilonGreedy(learner, 1, 0.95, 0.1)

        session.run(tf.initialize_all_variables())

        current_state = (0, 0)
        cumulative_reward = 0
        episode_t = 0
        for t in range(500):
            action_ = eps_greedy.get_action(state_index(current_state))
            next_state_ = transition(current_state, action_)
            reward_ = reward_function(next_state_)
            learner.update(state_index(current_state), action_, reward_, state_index(next_state_))
            eps_greedy.update()
            cumulative_reward += discount_factor ** episode_t * reward_
            episode_t += 1
            if next_state_[0] == num_actions and next_state_[1] == num_actions:
                current_state = (0, 0)

                print("Finished episode in t=%d - reward:%f" % (t, cumulative_reward))
                cumulative_reward = 0
                episode_t = 0
            else:
                current_state = next_state_
