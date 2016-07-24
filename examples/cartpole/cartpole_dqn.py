import collections
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import algorithms.dqn as dqn


class QNetwork:
    def __init__(self):
        self.state_dim = [4]
        self.discretized_actions = [0, 1]

    def build_network(self, state, reuse):
        input_size = self.state_dim[0]
        hidden1_size = 16
        hidden2_size = 16
        output_size = len(self.discretized_actions)

        hidden1_W = tf.get_variable("hidden1_W", shape=[input_size, hidden1_size],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=1/input_size))
        hidden1_b = tf.get_variable("hidden1_b", shape=[hidden1_size], initializer=tf.constant_initializer(0.))
        hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(state, hidden1_W), hidden1_b), name="hidden1")

        hidden2_W = tf.get_variable("hidden2_W", shape=[hidden1_size, hidden2_size],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=1/hidden1_size))
        hidden2_b = tf.get_variable("hidden2_b", shape=[hidden2_size], initializer=tf.constant_initializer(0.))
        hidden2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden1, hidden2_W), hidden2_b), name="hidden2")

        output_W = tf.get_variable("output_W", shape=[hidden2_size, output_size],
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=1/hidden2_size))
        output_b = tf.get_variable("output_b", shape=[output_size], initializer=tf.constant_initializer(0))

        return tf.nn.bias_add(tf.matmul(hidden2, output_W), output_b, name="Q")


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            if len(sys.argv) < 2:
                print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
                sys.exit(1)

            summary_dir = sys.argv[1] + '/summaries'

            env = gym.make("CartPole-v0")
            env.monitor.start(sys.argv[1] + '/monitor', force=True)

            network = QNetwork()
            global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0.), trainable=False)
            learner = dqn.DQN(network,
                              optimizer=tf.train.AdagradOptimizer(1e-1),
                              update_interval=1,
                              freeze_interval=100,
                              discount_factor=0.9,
                              experience_replay_memory=dqn.UniformExperienceReplayMemory(network.state_dim, 10000, 30),
                              double_dqn=True,
                              create_summaries=True,
                              global_step=global_step)
            exploration = dqn.EpsilonGreedy(learner, 0)
            session.run(tf.initialize_all_variables())

            os.mkdirs = summary_dir
            summary_writer = tf.train.SummaryWriter(summary_dir, session.graph)

            last_100 = collections.deque(maxlen=100)
            for episode in range(10000):
                state = env.reset()
                cumulative_reward = 0
                while True:
                    action = exploration.get_action(state)
                    next_state, reward, is_terminal, _ = env.step(action)
                    if is_terminal:
                        learner.update(state, action, next_state, -100, True)
                        learner.add_summaries(summary_writer, episode)
                        break
                    else:
                        learner.update(state, action, next_state, reward, False)
                        exploration.update()

                        cumulative_reward += reward
                        state = next_state
                last_100.append(cumulative_reward)
                last_100_mean = np.mean(last_100)
                print("Episode %d: %f(%f)" % (episode, last_100_mean, exploration._epsilon))
                if last_100_mean > 195:
                    break

