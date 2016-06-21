import tensorflow as tf
import sys
import gym
from q_network import QNetwork
from algorithms.dqn import DQN

with tf.Session() as session:
    q_network = QNetwork()
    learner = DQN(q_network, 0.5, 0.1, 1)  # TODO: check discount factor = 1

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
            action = learner.get_action(state)
            next_state, reward, is_terminal, _ = env.step(action)
            learner.update(state, action, reward, next_state)
            state = next_state
            cumulative_reward += reward
            if is_terminal:
                break
        print("Episode %d: %f" % (i, cumulative_reward))
    env.monitor.close()
