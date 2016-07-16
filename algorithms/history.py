import util.ring_buffer
import numpy as np
import algorithms.dqn as dqn


class HistoryStateManager:
    def __init__(self, history_length, buffer_size, state_dim, file_path=None):
        self._history_length = history_length - 1
        self._state_buffer = util.ring_buffer.RingBuffer(buffer_size, state_dim, file_path=file_path)
        self._valid = False
        self._episode_counter = 0
        self._counter = 0
        self._buffer_size = buffer_size

    def new_episode(self):
        self._episode_counter = 0
        self._valid = True

    def new_state(self, state):
        self._state_buffer.append(state)
        state_id = np.array([self._counter, self._episode_counter])
        self._counter += 1
        self._episode_counter += 1
        return state_id

    def retrieve_histories(self, state_ids, next_state_ids=None):
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

    @property
    def state_id_dim(self):
        return 2

    def create_replay_memory(self, mini_batch_size):
        return dqn.UniformExperienceReplayMemory(self.state_id_dim, self._buffer_size, mini_batch_size)


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
