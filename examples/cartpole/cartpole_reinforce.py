import sys

import gym

import algorithms.theano_backend.policy as policy
import algorithms.policy_gradient as pg
import util.gym_env
import lasagne
import functools

state_dim = 4
num_actions = 2
discount_factor = 1
optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=0.01)


def build_policy_network():
    l_in = lasagne.layers.InputLayer((None, state_dim))
    l_out = lasagne.layers.DenseLayer(l_in, num_actions, nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.HeUniform())
    return functools.partial(lasagne.layers.get_output, l_out), lasagne.layers.get_all_params(l_out)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
        sys.exit(1)

    env = gym.make("CartPole-v0")

    policy = policy.DiscreteTensorPolicy(num_actions, build_policy_network)
    learner = pg.REINFORCE(policy=policy, num_sample_episodes=5, optimizer=optimizer)

    util.gym_env.main_loop(env, learner, 200, num_iterations=100000, summary_frequency=None)


