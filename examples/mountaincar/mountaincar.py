import tensorflow as tf
import numpy as np
import sys
import gym
from q_network import build_q_network
import algorithms.dqn as dqn
import collections

with tf.device('/cpu:0'):
    with tf.Session() as session:
        state_dim = [2]
        learner = dqn.DQN(
            network_builder=build_q_network,
            state_dim=state_dim,
            num_actions=3,
            optimizer=tf.train.RMSPropOptimizer(1e-3, decay=0.9, momentum=0, epsilon=0.01),
            update_interval=1,
            freeze_interval=20,
            discount_factor=0.99,
            exploration=dqn.EpsilonGreedy(0.0, 0.99, 0.0),
            experience_replay_memory=dqn.UniformExperienceReplayMemory(state_dim, 100000, 200),
            minimum_memory_size=1000,
            td_rule='q-learning',
            create_summaries=False)

        session.run(tf.initialize_all_variables())

        if len(sys.argv) < 2:
            print("mountaincar.py takes one argument: the output directory of openai gym monitor data")
            sys.exit(1)

        env = gym.make("MountainCar-v0")
        #env.monitor.start(sys.argv[1], force=True)
        last_100 = collections.deque(maxlen=100)
        for i in range(3000):
            state = env.reset()
            cumulative_reward = 0
            for t in range(200):
                #env.render(mode='rgb_array')
                action = learner.get_action(state)
                next_state, reward, is_terminal, _ = env.step(action)
                learner.update(state, action, next_state, (reward + 11) if is_terminal else reward, is_terminal)
                state = next_state
                cumulative_reward += reward
                if is_terminal:
                    break
            last_100.append(cumulative_reward)
            print("Episode %d: %f - %f" % (i, cumulative_reward, np.mean(last_100)))
        env.monitor.close()
