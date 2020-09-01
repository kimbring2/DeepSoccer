import numpy as np
import random
from ForgER.sum_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, size, env_dict, alpha=0.6, eps=1e-6):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        #super(PrioritizedReplayBuffer, self).__init__(size, env_dict)
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.env_dict = env_dict

        assert alpha >= 0
        self._alpha = alpha
        self._eps = eps

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def get_stored_size(self):
        return len(self._storage)

    def get_buffer_size(self):
        return self._maxsize

    def add(self, kwargs):
        #print("kwargs: " + str(kwargs))

        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        #super().add(kwargs=kwargs)
        data = kwargs
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)

        return res

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            Priority level
        Returns
        -------
        encoded_sample: dict of np.array
            Array of shape(batch_size, ...) and dtype np.*32
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        idxes = self._sample_proportional(batch_size)
        it_sum = self._it_sum.sum()
        it_min = self._it_min.min()
        p_min = it_min / it_sum
        max_weight = (p_min * len(self._storage)) ** (- beta)
        p_sample = np.array([self._it_sum[idx] / it_sum for idx in idxes])
        weights = (p_sample*len(self._storage)) ** (- beta)
        weights = weights / max_weight
        encoded_sample = self._encode_sample(idxes)
        encoded_sample['weights'] = weights
        encoded_sample['indexes'] = idxes

        return encoded_sample

    @property
    def first_transition(self):
        return self._storage[0]

    def _encode_sample(self, idxes):
        batch = {key: list() for key in self.first_transition.keys()}
        #print("batch: " + str(batch))
        for i in idxes:
            data = self._storage[i]
            for key, value in data.items():
                batch[key].append(np.array(value))

        for key, value in batch.items():
            #print("type(key): " + str(type(key)))
            #print("self.env_dict[key]: " + str(self.env_dict[key]))
            #print("len(value): " + str(len(value)))
            #print("")
            #test = self.env_dict[key]
            test = np.array(value, dtype=self.env_dict[key])
            batch[key] = np.array(value, dtype=self.env_dict[key])

        return batch

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority >= 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + self._eps) ** self._alpha
            self._it_min[idx] = (priority + self._eps) ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class AggregatedBuff:
    def __init__(self, size, env_dict, episodes_to_decay=50, min_demo_proportion=0.0):
        buffer_base = PrioritizedReplayBuffer
            
        self._maxsize = size

        #print("size: " + str(size))
        #print("env_dict: " + str(env_dict))
        '''
        env_dict: {'n_reward': {'dtype': 'float32'}, 'to_demo': {'dtype': 'float32'}, 'demo': {'dtype': 'float32'}, 'n_done': {'dtype': 'bool'}, 
        'actual_n': {'dtype': 'float32'}, 'infrared': {'dtype': dtype('int64'), 'shape': ()}, 'next_infrared': {'dtype': dtype('int64'), 'shape': ()}, 
        'next_camera': {'dtype': dtype('float32'), 'shape': (512, 512, 3)}, 'done': {'dtype': 'bool'}, 'next_state': {'dtype': 'float32'}, 
        'n_lidar': {'dtype': dtype('float32'), 'shape': (1,)}, 'n_camera': {'dtype': dtype('float32'), 'shape': (512, 512, 3)}, 'next_lidar': {'dtype': dtype('float32'), 
        'shape': (1,)}, 'lidar': {'dtype': dtype('float32'), 'shape': (1,)}, 'n_infrared': {'dtype': dtype('int64'), 'shape': ()}, 'state': {'dtype': 'float32'}, 
        'camera': {'dtype': dtype('float32'), 'shape': (512, 512, 3)}, 'n_state': {'dtype': 'float32'}, 'action': {'dtype': 'int32', 'shape': 1}, 
        'reward': {'dtype': 'float32'}}
        '''

        self.demo_kwargs = {'size': size, 'env_dict': env_dict, 'eps': 1.0}
        self.demo_buff = buffer_base(size=size, env_dict=env_dict, eps=1.0)
        self.replay_buff = buffer_base(size=size, env_dict=env_dict, eps=1e-3)
        self.episodes_to_decay = episodes_to_decay
        self.episodes_done = 0
        self.min_demo_proportion = min_demo_proportion

    def add(self, kwargs, to_demo=0):
        #print("kwargs.keys(): " + str(kwargs.keys()))

        if to_demo:
            self.demo_buff.add(kwargs)
        else:
            #print("kwargs: " + str(kwargs))
            self.replay_buff.add(kwargs)
            if kwargs['done']:
                self.episodes_done += 1
                if self.demo_buff.get_stored_size() > 0\
                        and self.episodes_done > self.episodes_to_decay\
                        and self.min_demo_proportion == 0.:
                    self.demo_buff = PrioritizedReplayBuffer(self.demo_kwargs)

    def free_demo(self):
        self.demo_buff = PrioritizedReplayBuffer(self.demo_kwargs)

    @property
    def proportion(self):
        if self.episodes_to_decay == 0:
            proportion = 1. - self.min_demo_proportion
        else:
            proportion = min(1. - self.min_demo_proportion, self.episodes_done / self.episodes_to_decay)

        proportion = max(proportion, float(self.demo_buff.get_stored_size() == 0))

        return proportion

    def sample(self, n=32, beta=0.4):
        #print("n: " + str(n))
        agent_n = int(n * self.proportion)
        demo_n = n - agent_n
        
        if demo_n > 0 and agent_n > 0:
            demo_samples = self.demo_buff.sample(demo_n, beta)
            replay_samples = self.replay_buff.sample(agent_n, beta)
            samples = {key: np.concatenate((replay_samples[key], demo_samples[key]))
                       for key in replay_samples.keys()}
        elif agent_n == 0:
            samples = self.demo_buff.sample(demo_n, beta)
        else:
            samples = self.replay_buff.sample(agent_n, beta)

        samples = {key: np.squeeze(value) for key, value in samples.items()}

        return samples

    def update_priorities(self, indexes, priorities):
        n = len(indexes)
        agent_n = int(n * self.proportion)
        demo_n = n - agent_n
        if demo_n != 0:
            self.demo_buff.update_priorities(indexes[agent_n:], priorities[agent_n:])

        if agent_n != 0:
            self.replay_buff.update_priorities(indexes[:agent_n], priorities[:agent_n])

    def get_stored_size(self):
        return self.replay_buff.get_stored_size()

    def get_buffer_size(self):
        return self._maxsize
