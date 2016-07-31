width = 5
height = 5
num_actions = 4

reward_grid = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 10]]

discount_factor = 0.99
learning_rate = 1


def transition(state, action):
    if action == 0 and state[0] > 0:
        return state[0] - 1, state[1]
    elif action == 1 and state[0] < width - 1:
        return state[0] + 1, state[1]
    elif action == 2 and state[1] > 0:
        return state[0], state[1] - 1
    elif action == 3 and state[1] < height - 1:
        return state[0], state[1] + 1
    return state


def reward_function(state):
    assert state[0] >= 0
    assert state[0] < width
    assert state[1] >= 0
    assert state[1] < height
    return reward_grid[state[1]][state[0]]

def is_terminal(state):
    return reward_function(state) > 0
