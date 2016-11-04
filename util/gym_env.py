import sys
import collections
import numpy as np
import tqdm
import glob
import os
import re
import shutil


def main_loop(env, learner, num_time_steps, reward_horizon=100, reward_threshold=None, num_iterations=10000,
              enable_monitor=False,
              summary_frequency=100, do_render=False, save_model_directory=None, save_model_frequency=10,
              save_model_horizon=10, restore=False):
    sys.setrecursionlimit(50000)

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
    for episode in trange:
        state = env.reset()
        cumulative_reward = 0
        t = 0
        while num_time_steps is None or t < num_time_steps:
            t += 1
            if do_render:
                env.render()
            action = learner.get_action(state)
            next_state, reward, is_terminal, _ = env.step(action)
            update_result = learner.update(state, action, next_state, reward, is_terminal)
            cumulative_reward += reward
            if is_terminal:
                break
            state = next_state
        reward_window.append(cumulative_reward)
        reward_sma = np.mean(reward_window)
        trange.set_description("Episode %d: %f - %r" % (episode, reward_sma, update_result))
        if reward_threshold is not None and len(reward_window) == 100 and np.mean(reward_window) > reward_threshold:
            break

        if save_model_directory is not None and episode != start_iteration and episode % save_model_frequency == 0:
            open('%s/__backup_episode_%d' % (save_model_directory, episode), 'w').close()
            learner.save(save_model_directory, "backup_%d" % ((episode//save_model_frequency) % save_model_horizon))

        # if summary_frequency is not None and episode % summary_frequency == summary_frequency - 1:
        #     learner.add_summaries(summary_writer, episode)
