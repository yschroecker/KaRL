import tensorflow as tf
import numpy as np
import gym
import sys
import collections
import algorithms.policy_gradient as pg

state_dim = [4]
num_actions = 2
discount_factor = 1
optimizer = tf.train.RMSPropOptimizer(0.1)


def build_policy_network(state):
    weights = tf.get_variable("weights", shape=[state_dim[0], num_actions],
                              initializer=tf.truncated_normal_initializer())

    return tf.nn.softmax(tf.matmul(state, weights))

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        with tf.Session() as session:
            if len(sys.argv) < 2:
                print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
                sys.exit(1)

            summary_dir = sys.argv[1] + '/summaries'

            env = gym.make("CartPole-v0")
            #env.monitor.start(sys.argv[1] + '/monitor', force=True)

            policy = pg.DiscretePolicy(state_dim, num_actions, build_policy_network, optimizer)
            learner = pg.REINFORCE(policy=policy, state_dim=state_dim, discount_factor=discount_factor,
                                   optimizer=optimizer, num_sample_episodes=5)

            session.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter(summary_dir, session.graph)

            last_100 = collections.deque(maxlen=100)
            for episode in range(10000):
                state = env.reset()
                cumulative_reward = 0
                for t in range(200):
                    action = learner.get_action(state)
                    next_state, reward, is_terminal, _ = env.step(action)
                    statistics = learner.update(state, action, next_state, reward, is_terminal)
                    cumulative_reward += reward
                    if is_terminal:
                        break
                    state = next_state
                last_100.append(cumulative_reward)
                last_100_mean = np.mean(last_100)
                print("Episode %d: %f" % (episode, last_100_mean))
                if len(last_100) == 100 and last_100_mean > 195:
                    break
