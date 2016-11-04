import collections
import sys
import threading

import gym
import numpy as np
import tensorflow as tf

import algorithms.async as async
import algorithms.tensorflow_backend.policy_gradient as pg

state_dim = [4]
num_actions = 2
discount_factor = 1
num_instances = 10


def build_value_network(state, reuse):
    hidden1_size = 80

    hidden1_W = tf.get_variable("hidden1_W", shape=[state_dim[0], hidden1_size],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=2/np.sqrt(state_dim[0])))
    hidden1_b = tf.get_variable("hidden1_b", shape=[hidden1_size], initializer=tf.constant_initializer(0.))
    hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(state, hidden1_W), hidden1_b), name="hidden1")

    output_W = tf.get_variable("output_W", shape=[hidden1_size, 1],
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=2/np.sqrt(hidden1_size)))
    output_b = tf.get_variable("output_b", shape=[1], initializer=tf.constant_initializer(0))

    return tf.nn.bias_add(tf.matmul(hidden1, output_W), output_b, name="Q")


def build_policy_network(state):
    weights = tf.get_variable("weights", shape=[state_dim[0], num_actions],
                              initializer=tf.truncated_normal_initializer(stddev=1))

    return tf.nn.softmax(tf.matmul(state, weights))


def build_algorithm():
    actor_optimizer = tf.train.RMSPropOptimizer(1e-1/num_instances, decay=0.8, epsilon=0.01)
    critic_optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.995, epsilon=0.01)
    policy = pg.DiscretePolicy(state_dim, num_actions, build_policy_network, actor_optimizer, suffix='')
    return async.A3C(policy=policy, value_network_builder=build_value_network,
                     state_dim=state_dim, discount_factor=discount_factor,
                     steps_per_update=1, td_rule='1-step',
                     actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)


class CartpoleInstance:
    def __init__(self, learner, env, session, instance_id):
        self._learner = learner
        self._env = env
        self._instance_id = instance_id
        self._session = session

    def __call__(self):
        with self._session.as_default():
            last_100 = collections.deque(maxlen=100)
            for episode in range(10000//num_instances):
                state = self._env.reset()
                cumulative_reward = 0
                for t in range(200):
                    action = self._learner.get_action(self._instance_id, state)
                    next_state, reward, is_terminal, _ = self._env.step(action)
                    cumulative_reward += reward
                    if is_terminal:
                        self._learner.update(self._instance_id, state, action, next_state, -100, is_terminal)
                        break
                    else:
                        self._learner.update(self._instance_id, state, action, next_state, reward, is_terminal)

                        cumulative_reward += reward
                    state = next_state
                last_100.append(cumulative_reward)
                last_100_mean = np.mean(last_100)
                print("Episode %d: %f" % (episode, last_100_mean))
                if len(last_100) == 100 and last_100_mean > 195:
                    break


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        session = tf.InteractiveSession()
        if len(sys.argv) < 2:
            print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
            sys.exit(1)

        summary_dir = sys.argv[1] + '/summaries'

        #env.monitor.start(sys.argv[1] + '/monitor', force=True)

        learner = async.Async(algorithm=build_algorithm, num_instances=num_instances)
        instances = [CartpoleInstance(learner, gym.make("CartPole-v0"), session, i) for i in range(num_instances)]
        session.run(tf.initialize_all_variables())

        threads = [threading.Thread(target=instance) for instance in instances]
        for thread in threads:
            thread.start()

