import sys
import cv2
import logging
import gym
import tqdm
import functools
import numpy as np
import algorithms.dqn as dqn
import algorithms.history as history
import util.gym_env
import theano
import theano.tensor as T
import lasagne


def build_network(num_actions):
    l_in = lasagne.layers.InputLayer((None, 4, 84, 84))
    l_conv1 = lasagne.layers.Conv2DLayer(l_in, 32, 8, 4, b=lasagne.init.Constant(0.1))
    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1, 64, 4, 2, b=lasagne.init.Constant(0.1))
    l_conv3 = lasagne.layers.Conv2DLayer(l_conv2, 64, 3, 1, b=lasagne.init.Constant(0.1))
    l_dense1 = lasagne.layers.DenseLayer(l_conv3, 512, b=lasagne.init.Constant(0.1))
    l_dense2 = lasagne.layers.DenseLayer(l_dense1, num_actions,
                                         nonlinearity=lasagne.nonlinearities.linear, b=lasagne.init.Constant(0.1))
    return functools.partial(lasagne.layers.get_output, l_dense2), lasagne.layers.get_all_params(l_dense2)


def transform_state(state):
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)/255., (84, 84))


class EnvWrapper:
    def __init__(self, env):
        self._env = env

    @property
    def monitor(self):
        return self._env.monitor

    def reset(self):
        return transform_state(self._env.reset())

    def step(self, action):
        next_state, reward, is_terminal, info = self._env.step(action)
        return transform_state(next_state), min(1, reward), is_terminal, info


def init_live(gym_env):
    gym_env.step(1)
    for i in range(np.random.randint(30)):
        gym_env.step(0)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("breakout_dqn.py takes two argument: the output directory of openai gym monitor data and the"
              "output directory for backup data")
        sys.exit(1)

    gym_env = gym.make('Breakout-v0')

    summary_dir = sys.argv[1] + '/summaries'

    learner, env = history.create_dqn_with_history(
        network_builder=functools.partial(build_network, gym_env.action_space.n),
        state_dim=[84, 84],
        num_actions=gym_env.action_space.n,
        environment=EnvWrapper(gym_env),
        optimizer=functools.partial(lasagne.updates.rmsprop, learning_rate=0.0005, rho=0.95, epsilon=0.01),
        update_interval=1,
        replay_memory_type=dqn.UniformExperienceReplayMemory,
        freeze_interval=10000,
        discount_factor=0.99,
        minimum_memory_size=10000,
        td_rule='q-learning',
        history_length=4,
        mini_batch_size=32,
        buffer_size=900000,
        loss_clip_threshold=1,
        exploration=dqn.EpsilonGreedy(initial_epsilon=1,
                                      epsilon_decay=1e-6,
                                      min_epsilon=0.1,
                                      decay_type='linear'),
        create_summaries=False)

    test_mode = len(sys.argv) >= 3

    if test_mode:
        gym_env.monitor.start(sys.argv[1] + '/monitor', force=True, video_callable=lambda _: True)
    #else
        #gym_env.monitor.start(sys.argv[1] + '/monitor', force=True)

    t = 0
    if test_mode:
        tqdm_range = tqdm.trange(100)
    else:
        tqdm_range = tqdm.trange(500000)
    logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.ERROR)

    util.gym_env.main_loop(env, learner, None, 1000, num_iterations=10000000, enable_monitor=False,
                           save_model_directory=sys.argv[2], save_model_frequency=50)

