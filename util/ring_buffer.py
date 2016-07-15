import numpy as np


class RingBuffer:
    def __init__(self, capacity, element_dim):
        self._ringbuffer = np.zeros(np.hstack([capacity, element_dim]))
        self._head = 0
        self._size = 0
        self._elements_inserted = 0

    def append(self, elem):
        self._ringbuffer[self._head, :] = elem
        self._head = (self._head + 1) % self._ringbuffer.shape[0]
        self._size = min(self._size + 1, self._ringbuffer.shape[0])
        self._elements_inserted += 1

    def sample(self, num_samples):
        indices = np.random.choice(self._size, num_samples)
        return self._ringbuffer[indices, :]

    def __getitem__(self, item):
        if type(item[0]) is slice:
            start, end, step = item[0].indices(self._elements_inserted)
            wrapped_start = start % self._size
            wrapped_end = ((end - 1) % self._size) + 1
            if end > start and wrapped_start >= wrapped_end:
                return np.vstack([self._ringbuffer[wrapped_start::step, item[1]],
                                  self._ringbuffer[:wrapped_end:step, item[1]]])
            else:
                return self._ringbuffer[wrapped_start:wrapped_end:step, item[1]]
        else:
            return self._ringbuffer[item[0] % self._size, item[1]]


class RingBufferCollection:
    def __init__(self, capacity, element_dims):
        self._buffers = [RingBuffer(capacity, element_dim) for element_dim in element_dims]
        self._size = 0
        self._capacity = capacity

    def append(self, *elements):
        for element, buffer in zip(elements, self._buffers):
            buffer.append(element)
        self._size = min(self._size + 1, self._capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self._size, num_samples)
        return (buffer[indices, :] for buffer in self._buffers)
