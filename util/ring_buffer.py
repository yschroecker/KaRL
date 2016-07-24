import numpy as np
import h5py
import uuid


class RingBuffer:
    def __init__(self, capacity, element_dim, file_path=None, dtype=np.float16):
        self._use_disk = file_path is not None
        if self._use_disk:
            file = h5py.File(file_path)
            self._ringbuffer = file.create_dataset(str(uuid.uuid4()), np.hstack([capacity, element_dim]), dtype='f')
        else:
            self._ringbuffer = np.zeros(np.hstack([capacity, element_dim]), dtype=dtype)
        self._head = 0
        self.size = 0
        self._elements_inserted = 0

    def append(self, elem):
        self._ringbuffer[self._head, :] = elem
        self._head = (self._head + 1) % self._ringbuffer.shape[0]
        self.size = min(self.size + 1, self._ringbuffer.shape[0])
        self._elements_inserted += 1

    def sample(self, num_samples):
        assert(num_samples <= self.size)
        indices = np.random.choice(self.size, num_samples, replace=False)
        if self._use_disk:
            indices = sorted(indices.tolist())
        return self._ringbuffer[indices, :]

    def __getitem__(self, item):
        if type(item[0]) is slice:
            start, end, step = item[0].indices(self._elements_inserted)
            wrapped_start = start % self.size
            wrapped_end = ((end - 1) % self.size) + 1
            if end > start and wrapped_start >= wrapped_end:
                return np.vstack([self._ringbuffer[wrapped_start::step, item[1]],
                                  self._ringbuffer[:wrapped_end:step, item[1]]])
            else:
                return self._ringbuffer[wrapped_start:wrapped_end:step, item[1]]
        else:
            return self._ringbuffer[item[0] % self.size, item[1]]


class RingBufferCollection:
    def __init__(self, capacity, element_dims, file_path=None, dtype=np.float16):
        self._buffers = [RingBuffer(capacity, element_dim, file_path, dtype) for element_dim in element_dims]
        self.size = 0
        self._capacity = capacity

    def append(self, *elements):
        for element, buffer in zip(elements, self._buffers):
            buffer.append(element)
        self.size = min(self.size + 1, self._capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self.size, num_samples)
        return (buffer[indices, :] for buffer in self._buffers)
