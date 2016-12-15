import sys
import collections
import numpy as np
import tqdm
import glob
import os
import re
import shutil
import scipy.stats as stats
import algorithms.theano_backend.bokehboard


def load_last(learner, model_directory, save_model_frequency, save_model_horizon):
    backups = glob.glob('%s/__backup_episode_*' % model_directory)
    episodes = [int(re.findall(r'\d+', backup)[-1]) for backup in backups]
    start_iteration = np.max(episodes)
    learner.load(model_directory, "backup_%d" % ((start_iteration//save_model_frequency) % save_model_horizon))
    return learner


def collect_trajectories(env, learner, num_time_steps, num_iterations, reward_horizon=100, silent=False,
                         discount_factor=1):
    states = []
    actions = []
    rewards = []
    trange = tqdm.trange(num_iterations, disable=silent)
    for episode in trange:
        state = env.reset()
        cumulative_reward = 0
        t = 0
        episode_states = [state]
        episode_actions = []
        while num_time_steps is None or t < num_time_steps:
            action = learner.get_action(state)
            state, reward, is_terminal, _ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            cumulative_reward += reward * discount_factor**t
            t += 1
            if is_terminal:
                break
        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(cumulative_reward)
        trange.set_description("Episode %d: %f +/- %f" % (episode, np.mean(rewards[-reward_horizon:]),
                                                          stats.sem(rewards[-reward_horizon:])))
    return states, actions, rewards


def main_loop(env, learner, num_time_steps, reward_horizon=100, reward_threshold=None, num_iterations=10000,
              enable_monitor=False, discount_factor=1,
              create_summaries=False, do_render=False, save_model_directory=None, save_model_frequency=10,
              save_model_horizon=10, save_models=True, restore=False, reset_every=1, episode_hooks=[],
              update_every=1):
    sys.setrecursionlimit(50000)

    if create_summaries:
        bokehboard = algorithms.theano_backend.bokehboard.Bokehboard()
        bokehboard.show()
        if len(sys.argv) < 2:
            print("%s takes one argument: the output directory of monitor and summaries data" % sys.argv[0])
            sys.exit(1)

        summary_dir = sys.argv[1] + '/summaries'
        if enable_monitor:
            env.monitor.start(sys.argv[1] + '/monitor', force=True)

    reward_window = collections.deque(maxlen=reward_horizon)
    if restore:
        assert save_model_directory is not None, "No directory specified for restore operation"
        backups = glob.glob('%s/__backup_episode_*' % save_model_directory)
        episodes = [int(re.findall(r'\d+', backup)[-1]) for backup in backups]
        start_iteration = np.max(episodes)
        learner.load(save_model_directory, "backup_%d" % ((start_iteration//save_model_frequency) % save_model_horizon))
    else:
        start_iteration = 0

    if save_model_directory is not None:
        if start_iteration == 0 and os.path.exists(save_model_directory):
            shutil.rmtree(save_model_directory)
        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)

    trange = tqdm.trange(start_iteration, num_iterations)
    if reset_every == -1 or restore:
        state = env.reset()
    for episode in trange:
        if episode % reset_every == 0:
            state = env.reset()
        for hook in episode_hooks:
            hook(learner, episode)
        cumulative_reward = 0
        total_t = 0
        t = 0
        while num_time_steps is None or t < num_time_steps:
            if create_summaries:
                bokehboard.update()
            t += 1
            total_t += 1
            if do_render:
                env.render()
            action = learner.get_action(state)
            next_state, reward, is_terminal, _ = env.step(action)
            if total_t % update_every == 0:
                learner.update(state, action, next_state, reward, is_terminal)
            cumulative_reward += reward * discount_factor**t
            if is_terminal:
                break
            state = next_state
        reward_window.append(cumulative_reward)
        reward_sma = np.mean(reward_window)
        trange.set_description("Episode %d: %f" % (episode, reward_sma))
        if reward_threshold is not None and len(reward_window) == 100 and np.mean(reward_window) > reward_threshold:
            break

        if save_model_directory is not None and episode != start_iteration and episode % save_model_frequency == 0\
                and save_models:
            open('%s/__backup_episode_%d' % (save_model_directory, episode), 'w').close()
            learner.save(save_model_directory, "backup_%d" % ((episode//save_model_frequency) % save_model_horizon))

        # if summary_frequency is not None and episode % summary_frequency == summary_frequency - 1:
        #     learner.add_summaries(summary_writer, episode)
