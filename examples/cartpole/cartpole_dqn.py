import collections
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import algorithms.dqn as dqn
import algorithms.history

history_length = 1
state_dim = [4]
num_actions = 2
optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.995, epsilon=0.01)
update_interval = 1
freeze_interval = 1
discount_factor = 0.9
exploration = dqn.EpsilonGreedy(0)
buffer_size = 10000000
mini_batch_size = 500
td_rule = 'double-q-learning'
create_summaries = True


def build_network(state, reuse):
    input_size = 4 * history_length
    hidden1_size = 80
    output_size = 2
    state = tf.reshape(state, [-1, input_size])

    hidden1_W = tf.get_variable("hidden1_W", shape=[input_size, hidden1_size],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=1/input_size))
    hidden1_b = tf.get_variable("hidden1_b", shape=[hidden1_size], initializer=tf.constant_initializer(0.))
    hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(state, hidden1_W), hidden1_b), name="hidden1")

    output_W = tf.get_variable("output_W", shape=[hidden1_size, output_size],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=1/hidden1_size))
    output_b = tf.get_variable("output_b", shape=[output_size], initializer=tf.constant_initializer(0))

    return tf.nn.bias_add(tf.matmul(hidden1, output_W), output_b, name="Q")


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            if len(sys.argv) < 2:
                print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
                sys.exit(1)

            summary_dir = sys.argv[1] + '/summaries'

            env = gym.make("CartPole-v0")
            #env.monitor.start(sys.argv[1] + '/monitor', force=True)

            if history_length > 1:
                learner, env = algorithms.history.create_dqn_with_history(
                    history_length=history_length,
                    network_builder=build_network,
                    environment=env,
                    state_dim=state_dim,
                    num_actions=num_actions,
                    optimizer=optimizer,
                    update_interval=update_interval,
                    freeze_interval=freeze_interval,
                    discount_factor=discount_factor,
                    exploration=exploration,
                    buffer_size=buffer_size,
                    mini_batch_size=mini_batch_size,
                    replay_memory_type=dqn.UniformExperienceReplayMemory,
                    td_rule=td_rule,
                    create_summaries=create_summaries)
            else:
                learner = dqn.DQN(
                    network_builder=build_network,
                    state_dim=state_dim,
                    num_actions=num_actions,
                    optimizer=optimizer,
                    update_interval=update_interval,
                    freeze_interval=freeze_interval,
                    discount_factor=discount_factor,
                    exploration=exploration,
                    experience_replay_memory=dqn.UniformExperienceReplayMemory(state_dim, buffer_size, mini_batch_size),
                    td_rule=td_rule,
                    create_summaries=create_summaries)

            session.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter(summary_dir, session.graph)

            last_100 = collections.deque(maxlen=100)
            for episode in range(10000):
                state = env.reset()
                cumulative_reward = 0
                for t in range(200):
                    action = learner.get_action(state)
                    next_state, reward, is_terminal, _ = env.step(action)
                    if is_terminal:
                        statistics = learner.update(state, action, next_state, -100, is_terminal)
                        learner.add_summaries(summary_writer, episode)
                        break
                    else:
                        statistics = learner.update(state, action, next_state, reward, is_terminal)

                        cumulative_reward += reward
                        state = next_state
                last_100.append(cumulative_reward)
                last_100_mean = np.mean(last_100)
                print("Episode %d: %f(%f)" % (episode, last_100_mean, statistics.epsilon))
                if len(last_100) == 100 and last_100_mean > 195:
                    break

