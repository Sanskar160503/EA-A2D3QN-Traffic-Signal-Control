# per_buffer.py — Prioritized Experience Replay (PER)
# Standard replay samples uniformly — PER samples transitions where the agent
# made the biggest prediction error (high TD error), so it learns faster
# from the most informative experiences.
#
# Drop-in replacement for the deque memory in agent_v3.py

import numpy as np


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1)
        self.data     = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write    = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Usage (replaces deque in agent):
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        self.memory.add(s, a, r, s2, done)
        batch, indices, weights = self.memory.sample(batch_size)
        self.memory.update_priorities(indices, td_errors)
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree        = SumTree(capacity)
        self.alpha       = alpha        # priority exponent (0 = uniform, 1 = full priority)
        self.beta        = beta_start   # IS weight exponent (anneals to 1.0)
        self.beta_frames = beta_frames
        self.frame       = 1
        self.epsilon     = 1e-5         # small constant so zero-error transitions still sampled
        self.max_prio    = 1.0

    def add(self, s, a, r, s2, done):
        self.tree.add(self.max_prio, (s, a, r, s2, done))

    def sample(self, batch_size):
        indices  = []
        weights  = []
        batch    = []
        segment  = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_frames)
        self.frame += 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total
        if min_prob == 0:
            min_prob = 1e-7

        for i in range(batch_size):
            s_val = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s_val)
            prob = priority / self.tree.total
            w    = (prob / min_prob) ** (-self.beta)
            weights.append(w)
            indices.append(idx)
            batch.append(data)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()   # normalize
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_prio = max(self.max_prio, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
