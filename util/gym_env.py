import sys
import tensorflow as tf
import collections
import numpy as np
import tqdm


def main_loop(env, learner, num_timesteps, num_iterations=10000, enable_monitor=False, summary_frequency=100,
              do_render=False):
    if len(sys.argv) < 2:
        print("%s takes one argument: the output directory of monitor and summaries data" % sys.argv[0])
        sys.exit(1)

    summary_dir = sys.argv[1] + '/summaries'
    summary_writer = tf.train.SummaryWriter(summary_dir, tf.get_default_session().graph)
    if enable_monitor:
        env.monitor.start(sys.argv[1] + '/monitor', force=True)

    last_100 = collections.deque(maxlen=100)
    trange = tqdm.trange(num_iterations)
    for episode in trange:
        state = env.reset()
        cumulative_reward = 0
        for t in range(num_timesteps):
            if do_render:
                env.render()
            action = learner.get_action(state)
            next_state, reward, is_terminal, _ = env.step(action)
            update_result = learner.update(state, action, next_state, reward, is_terminal)
            cumulative_reward += reward
            if is_terminal:
                break
            state = next_state
        last_100.append(cumulative_reward)
        last_100_mean = np.mean(last_100)
        trange.set_description("Episode %d: %f - %r" % (episode, last_100_mean, update_result))
        if len(last_100) == 100 and last_100_mean > 195:
            break
        if summary_frequency is not None and episode % summary_frequency == summary_frequency - 1:
            learner.add_summaries(summary_writer, episode)
