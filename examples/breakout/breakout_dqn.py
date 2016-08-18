import sys
import cv2
import os
import logging
import gym
import tqdm
import tflearn
import tensorflow as tf
import numpy as np
import algorithms.dqn as dqn
import algorithms.history as history
import functools


def build_network(num_actions, state, reuse):
    state = tf.transpose(state, [0, 2, 3, 1])

    if not reuse:
        tf.image_summary(tf.get_variable_scope().name + "/state 0", state[:, :, :, 0:1])
        tf.image_summary(tf.get_variable_scope().name + "/state 1", state[:, :, :, 1:2])
        tf.image_summary(tf.get_variable_scope().name + "/state 2", state[:, :, :, 2:3])
        tf.image_summary(tf.get_variable_scope().name + "/state 3", state[:, :, :, 3:4])

    conv1 = tflearn.conv_2d(state, 32, filter_size=8, strides=4, weights_init='uniform_scaling',
                            bias_init=tf.constant_initializer(0.1), activation='elu', name="conv1", reuse=None)
    conv2 = tflearn.conv_2d(conv1, 64, filter_size=4, strides=2, weights_init='uniform_scaling', bias_init=tf.constant_initializer(0.1),
                            activation='elu', name="conv2", reuse=None)
    conv3 = tflearn.conv_2d(conv2, 64, filter_size=3, strides=2, weights_init='uniform_scaling', bias_init=tf.constant_initializer(0.1),
                            activation='elu', name="conv3", reuse=None)
    dense1 = tflearn.fully_connected(conv3, 512, activation='elu', weights_init='uniform_scaling', bias_init=tf.constant_initializer(0.1),
                                     name="dense1", reuse=None)
    output = tflearn.fully_connected(dense1, num_actions, weights_init='uniform_scaling', bias_init=tf.constant_initializer(0.1),
                                     name="output", reuse=None)
    return output


def transform_state(state):
    # state = state[30:, 7:-7, :]
    # state = scipy.misc.toimage(state).convert('L')
    # return scipy.misc.imresize(state, (84, 84)).astype(np.float32)
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)/255., (84, 84))


class EnvWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        return transform_state(self._env.reset())

    def step(self, action):
        next_state, reward, is_terminal, info = self._env.step(action)
        return transform_state(next_state), reward, is_terminal, info

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

        gym_env = gym.make('Breakout-v0')

        summary_dir = sys.argv[1] + '/summaries'

        global_step = tf.get_variable('global_step', shape=[], initializer=tf.constant_initializer(0.), trainable=False)

        learner, env = history.create_dqn_with_history(
            network_builder=functools.partial(build_network, gym_env.action_space.n),
            state_dim=[84, 84],
            num_actions=gym_env.action_space.n,
            environment=EnvWrapper(gym_env),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, epsilon=0.01),
            update_interval=1,
            replay_memory_type=dqn.UniformExperienceReplayMemory,
            freeze_interval=10000,
            discount_factor=0.99,
            minimum_memory_size=10000,
            td_rule='double-q-learning',
            history_length=4,
            mini_batch_size=32,
            buffer_size=900000,
            global_step=global_step,
            loss_clip_threshold=1,
            exploration=dqn.EpsilonGreedy(initial_epsilon=1,
                                          epsilon_decay=1e-6,
                                          min_epsilon=0.1,
                                          decay_type='linear'),
            create_summaries=True)

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

        t = 0
        if test_mode:
            tqdm_range = tqdm.trange(100)
        else:
            tqdm_range = tqdm.trange(500000)
        logging.getLogger('gym.monitoring.video_recorder').setLevel(logging.ERROR)
        cumulative_reward_avg = 0
        max_reward = 0
        cumulative_num_steps = 0
        ema_beta = 0.99
        avg_q = 0
        frame_skip = 4
        updates = 0
        for episode in tqdm_range:
            state = env.reset()
            is_terminal = False
            init_live(gym_env)
            cumulative_reward = 0
            num_steps = 0
            episode_q = 0
            current_lives = gym_env.ale.lives()
            while not is_terminal:
                if test_mode:
                    action = learner.max_action(state)
                else:
                    action = learner.get_action(state)
                # action = env.action_space.sample()
                next_state, reward, is_terminal, _ = env.step(action)

                if not test_mode:
                    if is_terminal or gym_env.ale.lives() < current_lives:
                        episode_q += learner.update(state, action, next_state, -1, True).mean_q
                        current_lives = gym_env.ale.lives()
                        if not is_terminal:
                            init_live(gym_env)
                        updates += 1
                    elif num_steps % frame_skip == 0:
                        episode_q += learner.update(state, action, next_state, reward, False).mean_q
                        updates += 1

                cumulative_reward += reward
                state = next_state
                num_steps += 1
            #print("reward %f in episode %d after %d steps (%f)" % (cumulative_reward, episode, num_steps, exploration._epsilon))
            cumulative_reward_avg = ema_beta * cumulative_reward_avg + (1 - ema_beta) * cumulative_reward
            if not test_mode:
                avg_q = ema_beta * avg_q + (1 - ema_beta) * episode_q/updates
            if cumulative_num_steps // 50000 < (cumulative_num_steps + num_steps) // 50000:
                print()
            cumulative_num_steps += num_steps
            max_reward = max(max_reward, cumulative_reward)
            tqdm_range.set_description(u"R: %f, max(R): %f, Q: %f, t: %d, \u03B5: %f" % (cumulative_reward_avg, max_reward,
                                                                                         avg_q, cumulative_num_steps,
                                                                                         learner._exploration.epsilon))

            learner.add_summaries(summary_writer, episode)
            if episode % 100 == 0 and not test_mode:
                saver.save(session, "/home/yannick/tmp/%d.ckpt" % episode)

