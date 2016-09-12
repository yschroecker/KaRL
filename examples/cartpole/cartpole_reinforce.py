import tensorflow as tf
import gym
import sys
import algorithms.policy_gradient as pg
import util.gym_env


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

            env = gym.make("CartPole-v0")

            policy = pg.DiscretePolicy(state_dim, num_actions, build_policy_network, optimizer)
            learner = pg.REINFORCE(policy=policy, state_dim=state_dim, discount_factor=discount_factor,
                                   optimizer=optimizer, num_sample_episodes=5)

            session.run(tf.initialize_all_variables())

            util.gym_env.main_loop(env, learner, 200, num_iterations=1000, summary_frequency=None)


