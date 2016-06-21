from gridworld import *
import numpy as np


def max_action(q_table, state):
    q_values = [q_table[a][state[1]][state[0]] for a in range(num_actions)]
    return np.argmax(q_values)


def epsilon_greedy(q_table, state, epsilon):
    rand = np.random.rand()
    if rand > epsilon:
        return max_action(q_table, state)
    else:
        return np.random.random_integers(0, num_actions - 1)


def simple_q_learning(q_table, state, action, next_state, reward):
    q_table[action][state[1]][state[0]] = reward + discount_factor * q_table[
        max_action(q_table, next_state)][next_state[1]][next_state[0]]

if __name__ == '__main__':
    np.random.seed(0)
    current_q_table = np.ones((num_actions, width, height)) * 1

    current_state = (0, 0)
    cumulative_reward = 0
    epsilon_ = 0.1
    episode_t = 0
    for t in range(500):
        action_ = epsilon_greedy(current_q_table, current_state, epsilon_)
        next_state_ = transition(current_state, action_)
        reward_ = reward_function(next_state_)
        simple_q_learning(current_q_table, current_state, action_, next_state_, reward_)
        cumulative_reward += discount_factor ** episode_t * reward_
        epsilon_ = max(0.1, epsilon_ * 0.95)
        episode_t += 1
        if next_state_[0] == 4 and next_state_[1] == 4:
            current_state = (0, 0)

            print("Finished episode in t=%d - reward:%f, epsilon:%f" % (t, cumulative_reward, epsilon_))
            cumulative_reward = 0
            episode_t = 0
        else:
            current_state = next_state_
