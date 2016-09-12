import util.ring_buffer
import numpy as np
import algorithms.dqn as dqn
import pickle


class HistoryStateManager:
    """
    Saving the history as part of the true state generates states with a lot of overlap and uses an unnecessary amount
    of memory. This class takes each state (without history) and memorizes it, returning a history state.
    retrieve_histories can then be used to convert the history state to the true history of states.
    """
    def __init__(self, history_length, buffer_size, state_dim, file_path=None):
        """
        :param history_length:
            The length of the history
        :param buffer_size:
            The number of states to be memorized. Should be identical with the buffer_size of the replay memory
        :param state_dim:
            The dimensions of each state without history
        :param file_path:
            The file used to store the true state. Should be None if possible. keeps the states in memory if None.
        """
        self._history_length = history_length - 1
        self._state_buffer = util.ring_buffer.RingBuffer(buffer_size, state_dim, file_path=file_path)
        self._valid = False
        self._episode_counter = 0
        self._counter = 0
        self._buffer_size = buffer_size

    @staticmethod
    def _save_path(dir_path, name):
        return "%s/history_%s" % (dir_path, name)

    def save(self, dir_path, name):
        assert self._valid
        with open(self._save_path(dir_path, name), 'wb') as f:
            pickle.dump([self._history_length, self._episode_counter, self._counter, self._buffer_size], f)
        self._state_buffer.save(dir_path, "%s_history" % name)

    @classmethod
    def load(cls, dir_path, name):
        history_manager = cls.__new__(cls)
        with open(cls._save_path(dir_path, name), 'rb') as f:
            [history_manager._history_length, history_manager._episode_counter,
             history_manager._counter, history_manager._buffer_size] = pickle.load(f)

        history_manager._state_buffer = util.ring_buffer.RingBuffer.load(dir_path,  "%s_history" % name)
        return history_manager

    def new_episode(self):
        """
        Needs to be called at the beginning of each episode in order to not get histories across episodes
        """
        self._episode_counter = 0
        self._valid = True

    def new_state(self, state):
        """
        Creates a history state
        :param state: The true state to be memorized
        :return: The history state.
        """
        self._state_buffer.append(state)
        state_id = np.array([self._counter, self._episode_counter])
        self._counter += 1
        self._episode_counter += 1
        return state_id

    def retrieve_histories(self, state_ids, next_state_ids=None):
        """
        Convert history states to true states. Can be passed to DQN as state_preprocessor.
        :param state_ids:
            history states
        :param next_state_ids:
            history states following state_ids. Has to be state_ids + 1.
        :return:
            True history of states for state_ids and next_state_ids. Only returns history of states for state_ids if
            next_state_ids is None
        """
        state_ids = np.array(state_ids, dtype=np.int64)
        assert next_state_ids is None or np.all(next_state_ids == state_ids + 1)
        if state_ids.shape == (2,):
            state_ids = [state_ids]
            next_state_ids = [next_state_ids]

        histories = []
        next_histories = []
        for state_id in state_ids:
            counter, episode_counter = state_id
            state = self._state_buffer[counter - min(episode_counter, self._history_length):counter + 1, :]
            if self._history_length > episode_counter:
                state = np.concatenate([[state[0] for _ in range(self._history_length - episode_counter)], state])
            histories.append(state)
            if next_state_ids is not None and next_state_ids[0] is not None:
                next_histories.append(np.concatenate([state[1:], [self._state_buffer[counter + 1, :]]]))
        if next_state_ids is not None and next_state_ids[0] is not None:
            return histories, next_histories
        else:
            return histories

    def create_replay_memory(self, mini_batch_size, memory_type=dqn.UniformExperienceReplayMemory):
        """
        Creates a replay memory with mini_batch_size that stores history_states.
        :param mini_batch_size:
            see replay memory doc
        :param memory_type
            Class of the replay memory to be created
        :return:
            replay memory instance
        """
        return memory_type(self._state_id_dim, self._buffer_size, mini_batch_size, dtype=np.int32)

    @property
    def _state_id_dim(self):
        return 2


class EnvironmentWithHistory:
    """
    Convenience wrapper around gym-style environments. Manages calls to history_manager to create history states.
    Environments that are not OpenAI gym environments need to be wrapped in a class that provides a reset() which
    returns the intiial state and a step(action) method which returns a 4-tuple
    (next_state, reward, is_terminal, additional_info).
    """
    def __init__(self, environment, history_manager):
        """
        :param environment:
            The environment to be wrapped. Must define reset() -> initial_state and
            step(action) -> (next_state, reward, is_terminal, additional_info)
        :param history_manager: HistoryStateManager
            The history manager used with this environment.
        """
        self._env = environment
        self._history = history_manager

    def save(self, dir_path, name):
        self._history.save(dir_path, name)

    def load(self, dir_path, name):
        self._history = HistoryStateManager.load(dir_path, name)

    def reset(self):
        """
        Starts a new episode
        :return:
            The initial (history) state of the new episode
        """
        self._history.new_episode()
        return self._history.new_state(self._env.reset())

    def step(self, action):
        """
        Takes an action in the wrapped environment
        :param action:
            The action as you would pass it to step() of the true environment
        :return:
            next_state, reward, is_terminal, info where next_state is a history state
        """
        next_state, reward, is_terminal, info = self._env.step(action)
        next_state = self._history.new_state(next_state)
        return next_state, reward, is_terminal, info

    def retrieve_histories(self, state_ids, next_state_ids=None):
        return self._history.retrieve_histories(state_ids, next_state_ids)


def create_dqn_with_history(state_dim, environment, history_length, replay_memory_type, buffer_size, mini_batch_size,
                            file_path=None, *args, **kwargs):
    """
    Helper method to create DQN instance and wrapped environment with history
    :param state_dim:
        Dimensionality of the state without history. Must be list (not np.ndarray)
    :param environment:
        Environment to wrap without history (see EnvironmentWithHistory for requirements)
    :param history_length:
    :param replay_memory_type:
        class of the replay memory, e.g. UniformExperienceReplayMemory
    :param buffer_size:
        number of states to be remembered
    :param mini_batch_size:
    :param file_path:
        File to store the true states in. Keeps states in memory if passed None
    :param args:
        Arguments to pass to DQN
    :param kwargs:
        Keyworkd arguments to pass to DQN
    :return: (DQN, EnvironmentWithHistory)
    """
    history_manager = HistoryStateManager(history_length, buffer_size, state_dim, file_path)
    environment_with_history = EnvironmentWithHistory(environment, history_manager)
    return dqn.DQN(experience_replay_memory=history_manager.create_replay_memory(mini_batch_size, replay_memory_type),
                   state_preprocessor=environment_with_history.retrieve_histories,
                   state_dim=[history_length] + state_dim,
                   *args, **kwargs), environment_with_history


if __name__ == '__main__':
    history_manager = HistoryStateManager(3, 10, [1])

    states = []
    j = 0
    for i in range(8):
        if i % 4 == 0:
            states.append(history_manager.new_state(i))
            history_manager.new_episode()
            j += 10
        states.append(history_manager.new_state(i))
    states = states[1:]

    for i in range(8):
        if i != 4:
            print(history_manager.retrieve_histories(states[i], states[i + 1]))

    print("==================================")
    print(history_manager.retrieve_histories(states[:3], states[1:4]))
