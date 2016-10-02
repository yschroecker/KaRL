import theano
import theano.tensor as T
import lasagne
import gridworld
import algorithms.dqn as dqn
import functools


def build_network():
    l_in = lasagne.layers.InputLayer((None, gridworld.width * gridworld.height))
    l_out = lasagne.layers.DenseLayer(l_in, gridworld.num_actions, lasagne.init.Constant(10.), b=None)
    def build_instance(state):
        state = T.extra_ops.to_one_hot(T.cast(state, 'int32'), gridworld.width * gridworld.height)
        return lasagne.layers.get_output(l_out, state)
    return build_instance, lasagne.layers.get_all_params(l_out, trainable=True)

if __name__ == '__main__':
    learner = dqn.DQN(network_builder=build_network,
                      state_dim=1,
                      num_actions=gridworld.num_actions,
                      optimizer=functools.partial(lasagne.updates.sgd, learning_rate=0.5),
                      discount_factor=gridworld.discount_factor,
                      exploration=dqn.EpsilonGreedy(0.1),
                      experience_replay_memory=dqn.UniformExperienceReplayMemory(1))


    current_state = (0, 0)
    cumulative_reward = 0
    episode_t = 0
    for t in range(500):
        action_ = learner.get_action(gridworld.state_index(current_state))
        next_state_ = gridworld.transition(current_state, action_)
        reward_ = gridworld.reward_function(next_state_)
        cumulative_reward += gridworld.discount_factor ** episode_t * reward_
        episode_t += 1
        is_terminal = gridworld.is_terminal(next_state_)
        learner.update(gridworld.state_index(current_state), action_, gridworld.state_index(next_state_), reward_, is_terminal)
        if is_terminal:
            current_state = (0, 0)

            print("Finished episode in t=%d - reward:%f" % (t, cumulative_reward))
            cumulative_reward = 0
            episode_t = 0
        else:
            current_state = next_state_
