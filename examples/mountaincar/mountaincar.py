import tensorflow as tf
import sys
import gym
from q_network import QNetwork
import algorithms.dqn as dqn

with tf.device('/cpu:0'):
    with tf.Session() as session:
        q_network = QNetwork()
        learner = dqn.DQN(q_network,
                          learning_rate=0.01,
                          discount_factor=0.99,
                          update_interval=1
                          experience_replay_memory=dqn.UniformExperienceReplayMemory(2, 100000, 32))
        exploration = dqn.EpsilonGreedy(learner, 1, 0.99, 0.1)

        session.run(tf.initialize_all_variables())

        if len(sys.argv) < 2:
            print("mountaincar.py takes one argument: the output directory of openai gym monitor data")
            sys.exit(1)

        env = gym.make("MountainCar-v0")
        env.monitor.start(sys.argv[1], force=True)
        for i in range(3000):
            state = env.reset()
            cumulative_reward = 0
            for t in range(200):
                #env.render(mode='rgb_array')
                action = exploration.get_action(state)
                next_state, reward, is_terminal, _ = env.step(action)
                learner.update(state, action, next_state, reward)
                exploration.update()
                state = next_state
                cumulative_reward += reward
                if is_terminal:
                    break
            print("Episode %d: %f" % (i, cumulative_reward))
        env.monitor.close()
