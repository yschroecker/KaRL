import gym.envs.classic_control
import sys
import algorithms.theano_backend.policy as policy
import algorithms.theano_backend.temporal_difference as td
import algorithms.policy_gradient as pg
import functools
import theano
import theano.tensor as T
import lasagne
import util.gym_env


state_dim = [4]
num_actions = 2
actor_optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=1e-2)
td_optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=1e-3, rho=0.995, epsilon=1e-2)
discount_factor = 0.9
buffer_size = 10000000
td_rule = '1-step'


def build_value_network():
    l_in = lasagne.layers.InputLayer((None, 4))
    l_hidden = lasagne.layers.DenseLayer(l_in, 80, nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeUniform(gain='relu'))
    l_out = lasagne.layers.DenseLayer(l_hidden, 1, nonlinearity=lasagne.nonlinearities.linear,
                                      W=lasagne.init.HeUniform())
    return functools.partial(lasagne.layers.get_output, l_out), lasagne.layers.get_all_params(l_out)


def build_policy_network():
    l_in = lasagne.layers.InputLayer((None, state_dim[0]))
    l_out = lasagne.layers.DenseLayer(l_in, num_actions, nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.HeUniform())
    return functools.partial(lasagne.layers.get_output, l_out), lasagne.layers.get_all_params(l_out)


class Cartpole(gym.envs.classic_control.cartpole.CartPoleEnv):
    def step(self, action):
        state, reward, is_terminal, info = super().step(action)
        if is_terminal:
            reward = -100
        return state, reward, is_terminal, info

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
        sys.exit(1)

    env = Cartpole()

    policy = policy.DiscreteTensorPolicy(num_actions, build_policy_network)
    td_learner = td.TemporalDifferenceLearnerV(build_value_network, td_optimizer, state_dim, discount_factor,
                                               td_rule=td_rule, loss_clip_threshold=None, loss_clip_mode='None',
                                               create_summaries=False)
    learner = pg.SimpleActorCritic(policy, td_learner, discount_factor, actor_optimizer, 1)

    util.gym_env.main_loop(env, learner, 200, num_iterations=100000, summary_frequency=None)

