import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────
#  Prioritized Replay Buffer (PER)
# ─────────────────────────────────────────────

class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write     = 0

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
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000, alpha=0.4, beta_start=0.4, beta_frames=80000):
        # alpha=0.4 (was 0.6) — less aggressive priority, more uniform sampling diversity
        # this prevents the buffer from over-focusing on a narrow set of transitions
        self.tree        = SumTree(capacity)
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1
        self.epsilon     = 1e-4    # slightly larger — ensures all transitions have chance
        self.max_prio    = 1.0

    def add(self, s, a, r, s2, done):
        self.tree.add(self.max_prio, (s, a, r, s2, done))

    def sample(self, batch_size):
        indices, weights, batch = [], [], []
        segment  = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_frames)
        self.frame += 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / (self.tree.total + 1e-8)
        min_prob = max(min_prob, 1e-7)

        for i in range(batch_size):
            s_val = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s_val)
            prob = priority / (self.tree.total + 1e-8)
            weights.append((prob / min_prob) ** (-self.beta))
            indices.append(idx)
            batch.append(data)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority      = (abs(error) + self.epsilon) ** self.alpha
            self.max_prio = max(self.max_prio, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ─────────────────────────────────────────────
#  Dueling DQN Network
# ─────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features  = self.feature(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────
#  D3QN Agent
# ─────────────────────────────────────────────

class D3QNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        # FIX 1: faster decay — reaches 0.05 by ~ep 700 in a 1000-ep run
        # old: 0.998 → epsilon=0.15 at ep 1000 (never fully exploits)
        # new: 0.9935 → epsilon=0.05 at ep ~700
        self.epsilon_decay = 0.9935
        self.batch_size    = 256     # FIX 2: larger batch — smoother loss surface
        # FIX 3: slightly higher lr — 5e-5 was too slow to escape local optimum
        self.lr            = 1e-4

        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[D3QN] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[D3QN] GPU: {torch.cuda.get_device_name(0)}")

        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # FIX 4: LR scheduler — halves learning rate at ep 500 and 800
        # prevents oscillation in late training while keeping early learning fast
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[500 * 1000, 800 * 1000], gamma=0.5
        )

        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.update_counter     = 0
        # FIX 5: sync target net less often — keeps targets stable longer
        # old: 500 steps — too frequent, target net chases policy too closely
        # new: 1000 steps
        self.target_update_freq = 1000

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.policy_net(s)).item()

    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch, indices, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights     = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        rewards = torch.clamp(rewards, -10.0, 10.0)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        element_loss = self.loss_fn(q_values, target_q)
        loss         = (weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  [Target net synced @ step {self.update_counter}]")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
