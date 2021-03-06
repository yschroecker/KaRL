import numpy as np
import h5py
import uuid
import os


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

    def save(self, dir_path, name):
        if self._use_disk:
            return

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with h5py.File(dir_path + '/ringbuffers') as file:
            if name in file:
                del file[name]
            saved_data = file.create_dataset(name, data=self._ringbuffer)
            saved_data.attrs['size'] = self.size
            saved_data.attrs['head'] = self._head
            saved_data.attrs['elements_inserted'] = self._elements_inserted

    @classmethod
    def load(cls, dir_path, name):
        with h5py.File(dir_path + '/ringbuffers') as file:
            dataset = file[name]
            ring_buffer = cls.__new__(cls)
            ring_buffer._ringbuffer = dataset[()]
            ring_buffer.size = dataset.attrs['size']
            ring_buffer._head = dataset.attrs['head']
            ring_buffer._elements_inserted = dataset.attrs['elements_inserted']
            return ring_buffer

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

    def __repr__(self):
        return self._ringbuffer.__repr__()


class RingBufferCollection:
    def __init__(self, capacity, element_dims, *args, **kwargs):
        self._buffers = [RingBuffer(capacity, element_dim, *args, **kwargs) for element_dim in element_dims]
        self.size = 0
        self._capacity = capacity

    def append(self, *elements):
        for element, buffer in zip(elements, self._buffers):
            buffer.append(element)
        self.size = min(self.size + 1, self._capacity)

    def sample(self, num_samples):
        indices = np.random.choice(self.size, num_samples)
        return (buffer[indices, :] for buffer in self._buffers)

    def save(self, dir_path, name):
        with h5py.File(dir_path + '/ringbuffers') as file:
            if name in file:
                del file[name]
            group = file.create_group(name)
            group.attrs['num_buffers'] = len(self._buffers)
            group.attrs['capacity'] = self._capacity

        for i, buffer in enumerate(self._buffers):
            buffer.save(dir_path, '%s/%d' % (name, i))

    @classmethod
    def load(cls, dir_path, name):
        ring_buffer_collection = cls.__new__(cls)
        with h5py.File(dir_path + '/ringbuffers') as file:
            num_buffers = file[name].attrs['num_buffers']
            ring_buffer_collection._capacity = file[name].attrs['capacity']

        ring_buffer_collection._buffers = [RingBuffer.load(dir_path, '%s/%d' % (name, i)) for i in range(num_buffers)]
        ring_buffer_collection.size = ring_buffer_collection._buffers[0].size
        return ring_buffer_collection

    def __repr__(self):
        return '\n'.join([buffer.__repr__() for buffer in self._buffers])


if __name__ == '__main__':
    import shutil

    path = '/home/yannick/tmpout/savetest'
    shutil.rmtree(path, True)
    buffer = RingBuffer(2, 2)
    buffer.append([1, 1])
    buffer.append([2, 1])
    buffer.append([1, 2])
    buffer.save(path, 'a')
    buffer2 = buffer.load(path, 'a')
    print(buffer2)

    buffercollection = RingBufferCollection(2, [1, 2])
    buffercollection.append([1], [1, 1])
    buffercollection.append([2], [2, 1])
    buffercollection.append([3], [1, 2])
    buffercollection.save(path, 'b')
    buffercollection2 = RingBufferCollection.load(path, 'b')
    print(buffercollection2)
