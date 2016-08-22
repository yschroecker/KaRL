import sys
import cv2
import logging
import gym
import tqdm
import tflearn
import tensorflow as tf
import numpy as np
import algorithms.async as async
import algorithms.policy_gradient as pg
import algorithms.history as history
import functools
import collections


def build_value_network(num_actions, state, reuse):
    state = tf.transpose(state, [0, 2, 3, 1])

    if not reuse:
        tf.image_summary(tf.get_variable_scope().name + "/state 0", state[:, :, :, 0:1])
        tf.image_summary(tf.get_variable_scope().name + "/state 1", state[:, :, :, 1:2])
        tf.image_summary(tf.get_variable_scope().name + "/state 2", state[:, :, :, 2:3])
        tf.image_summary(tf.get_variable_scope().name + "/state 3", state[:, :, :, 3:4])

    conv1 = tflearn.conv_2d(state, 32, filter_size=8, strides=4, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1), activation='relu', name="conv1", reuse=None)
    conv2 = tflearn.conv_2d(conv1, 64, filter_size=4, strides=2, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1), activation='relu', name="conv2", reuse=None)
    conv3 = tflearn.conv_2d(conv2, 64, filter_size=3, strides=2, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1), activation='relu', name="conv3", reuse=None)
    dense1 = tflearn.fully_connected(conv3, 512, activation='relu', weights_init='uniform_scaling',
                                     bias_init=tf.constant_initializer(0.1), name="dense1", reuse=None)
    output = tflearn.fully_connected(dense1, num_actions, weights_init='uniform_scaling',
                                     bias_init=tf.constant_initializer(0.1), name="output", reuse=None)
    return output


def build_policy_network(num_actions, state, reuse):
    state = tf.transpose(state, [0, 2, 3, 1])

    conv1 = tflearn.conv_2d(state, 32, filter_size=8, strides=4, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1), activation='relu', name="conv1", reuse=None)
    conv2 = tflearn.conv_2d(conv1, 64, filter_size=4, strides=2, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1),
                            activation='relu', name="conv2", reuse=None)
    conv3 = tflearn.conv_2d(conv2, 64, filter_size=3, strides=2, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1),
                            activation='relu', name="conv3", reuse=None)
    dense1 = tflearn.fully_connected(conv3, 512, activation='relu', weights_init='uniform_scaling',
                                     bias_init=tf.constant_initializer(0.1),
                                     name="dense1", reuse=None)
    output = tflearn.fully_connected(dense1, num_actions, weights_init='uniform_scaling',
                                     bias_init=tf.constant_initializer(0.1), activation='softmax',
                                     name="output", reuse=None)
    return output


def build_algorithm():
    num_instances = 20
    state_dim = [4, 84, 84]
    actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0025, decay=0.95, epsilon=0.01),
    critic_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, epsilon=0.01),
    policy = pg.DiscretePolicy(state_dim, gym_env.action_space.n, build_policy_network, actor_optimizer)
    return async.A3C(policy=policy, value_network_builder=build_value_network,
                     state_dim=state_dim, discount_factor=0.99,
                     steps_per_update=50, td_rule='deepmind-n-step',
                     actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)


def transform_state(state):
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)/255., (84, 84))


class EnvWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        return transform_state(self._env.reset())

    def step(self, action):
        next_state, reward, is_terminal, info = self._env.step(action)
        return transform_state(next_state), reward, is_terminal, info


class BreakoutInstance:
    def __init__(self, learner, instance_id):
        self._gym_env = gym.make('Breakout-v0')
        self._learner = learner
        self._env = EnvWrapper(self._gym_env)
        self._instance_id = instance_id

    def __call__(self):
        tqdm_range = tqdm.trange(500000, position=self._instance_id)
        history = collections.deque(maxlen=4)
        next_history = collections.deque(maxlen=4)

        frame_count = 0
        history_length = 4
        frame_skip = 4
        for episode in tqdm_range:
            inital_state = self._env.reset()
            for _ in history_length:
                history.append(inital_state)
                next_history.append(inital_state)




def init_live(gym_env):
    gym_env.step(1)
    for i in range(np.random.randint(30)):
        gym_env.step(0)

if __name__ == '__main__':
    with tf.device('gpu:0'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)  # Allocate only 3GB due to GTX 970s weird memory
        session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        #session = tf.InteractiveSession()

        if len(sys.argv) < 2:
            print("breakout_dqn.py takes one argument: the output directory of openai gym monitor data")
            sys.exit(1)


        summary_dir = sys.argv[1] + '/summaries'

        global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0.), trainable=False)

        # learner, env = history.create_dqn_with_history(
        #     network_builder=functools.partial(build_network, gym_env.action_space.n),
        #     state_dim=[84, 84],
        #     num_actions=gym_env.action_space.n,
        #     environment=EnvWrapper(gym_env),
        #     optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, epsilon=0.01),
        #     update_interval=1,
        #     replay_memory_type=dqn.UniformExperienceReplayMemory,
        #     freeze_interval=10000,
        #     discount_factor=0.99,
        #     minimum_memory_size=10000,
        #     td_rule='q-learning',
        #     history_length=4,
        #     mini_batch_size=32,
        #     buffer_size=900000,
        #     global_step=global_step,
        #     loss_clip_threshold=1,
        #     exploration=dqn.EpsilonGreedy(initial_epsilon=1,
        #                                   epsilon_decay=1e-6,
        #                                   min_epsilon=0.1,
        #                                   decay_type='linear'),
        #     create_summaries=True)

        test_mode = len(sys.argv) >= 3

        if test_mode:
            gym_env.monitor.start(sys.argv[1] + '/monitor', force=True, video_callable=lambda _: True)
            #else
            #gym_env.monitor.start(sys.argv[1] + '/monitor', force=True)

        summary_writer = tf.train.SummaryWriter(summary_dir, session.graph)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver(max_to_keep=None)
        if test_mode:
            saver.restore(session, sys.argv[2])

        if test_mode:
            tqdm_range = tqdm.trange(100)
        else:
            tqdm_range = tqdm.trange(500000)
        logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.ERROR)
