import tensorflow as tf
from gridworld import *
import algorithms.dqn as dqn
import sys
import datetime


class PseudoQNetwork:
    def __init__(self):
        self.action_dim = []
        self.discretized_actions = list(range(num_actions))
        self.state_dim = []

    @staticmethod
    def build_network(states):
        states = tf.cast(states, tf.int64)
        state = tf.one_hot(states, width * height, 1, 0, axis=-1)
        input_dim = width * height
        state = tf.reshape(tf.cast(state, tf.float32), [-1, input_dim])

        weights = tf.get_variable("weights", shape=[input_dim, num_actions], initializer=tf.constant_initializer(1))

        return tf.matmul(state, weights)


def state_index(state):
    return state[0] * width + state[1]


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            q_network = PseudoQNetwork()
            learner = dqn.DQN(q_network, 1, discount_factor, dqn.UniformExperienceReplayMemory(1, 1), freeze_interval=1)
            eps_greedy = dqn.EpsilonGreedy(learner, 0.1, 0.95, 0.1)

            session.run(tf.initialize_all_variables())

            assert len(sys.argv) > 1
            summary_writer = tf.train.SummaryWriter(
                "%s/%s" % (sys.argv[1], datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')))

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

                '''
                summary_writer.add_summary(
                    session.run(tf.merge_all_summaries(), feed_dict={
                        learner._state: state_index(current_state), learner._action: action_,
                        learner._next_state: state_index(next_state_), learner._reward: reward_
                    }))
                    '''


