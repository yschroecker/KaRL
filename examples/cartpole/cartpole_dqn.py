import collections
import gym
import sys
import os
import numpy as np
import algorithms.dqn as dqn
import algorithms.history
import functools
import theano
import theano.tensor as T
import lasagne


history_length = 1
state_dim = [4]
num_actions = 2
optimizer = functools.partial(lasagne.updates.rmsprop, learning_rate=1e-3, rho=0.995, epsilon=1e-2)
update_interval = 1
freeze_interval = 1
discount_factor = 0.9
exploration = dqn.EpsilonGreedy(0)
buffer_size = 10000000
mini_batch_size = 500
td_rule = 'q-learning'
create_summaries = True

def build_network():
    l_in = lasagne.layers.InputLayer((None, 4 * history_length))
    l_hidden = lasagne.layers.DenseLayer(l_in, 80, nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeUniform(gain='relu'))
    l_out = lasagne.layers.DenseLayer(l_hidden, num_actions, nonlinearity=lasagne.nonlinearities.linear,
                                      W=lasagne.init.HeUniform())
    return functools.partial(lasagne.layers.get_output, l_out), lasagne.layers.get_all_params(l_out)

if __name__ == '__main__':
        if len(sys.argv) < 2:
            print("cartpole_dqn.py takes one argument: the output directory of openai gym monitor data")
            sys.exit(1)

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

        last_100 = collections.deque(maxlen=100)
        for episode in range(10000):
            state = env.reset()
            cumulative_reward = 0
            for t in range(200):
                action = learner.get_action(state)
                next_state, reward, is_terminal, _ = env.step(action)
                if is_terminal:
                    statistics = learner.update(state, action, next_state, -100, is_terminal)
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

