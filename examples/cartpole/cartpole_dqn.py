import collections
import gym.envs.classic_control
import sys
import os
import numpy as np
import algorithms.dqn as dqn
import algorithms.history
import functools
import theano
import theano.tensor as T
import lasagne
import util.gym_env


history_length = 1
state_dim = [4]
num_actions = 2
optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=1e-3, rho=0.995, epsilon=1e-2)
# optimizer = functools.partial(lasagne.updates.sgd, learning_rate=1e-4)
update_interval = 1
freeze_interval = 1
discount_factor = 0.9
exploration = dqn.EpsilonGreedy(0)
buffer_size = 10000000
mini_batch_size = 500
td_rule = 'q-learning'
create_summaries = False


def build_network():
    l_in = lasagne.layers.InputLayer((None, 4 * history_length))
    l_hidden = lasagne.layers.DenseLayer(l_in, 80, nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeUniform(gain='relu'), name='Q_hidden')
    l_out = lasagne.layers.DenseLayer(l_hidden, num_actions, nonlinearity=lasagne.nonlinearities.linear,
                                      W=lasagne.init.HeUniform(), name='Q_out')
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

        util.gym_env.main_loop(env, learner, restore=False,
                               num_time_steps=200,
                               reward_threshold=295,
                               create_summaries=create_summaries,
                               save_model_directory='/tmp/cartpole')  # TODO: remove backups for cartpole

