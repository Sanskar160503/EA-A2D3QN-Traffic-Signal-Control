# agent_v2.py — Stabilized DQN Agent
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # wider first layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # Better weight initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma           = 0.95       # slightly lower — less long-horizon noise
        self.epsilon         = 1.0
        self.epsilon_min     = 0.05
        self.epsilon_decay   = 0.998      # SLOWER decay — more exploration before converging
        self.batch_size      = 128        # larger batch — smoother gradient estimates
        self.lr              = 1e-4       # LOWER lr — prevents Q-value explosion

        self.memory          = deque(maxlen=20000)   # bigger buffer
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net      = DQN(state_dim, action_dim).to(self.device)
        self.target_net      = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer       = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn         = nn.SmoothL1Loss()   # Huber loss — robust to reward outliers

        self.update_counter       = 0
        self.target_update_freq   = 500   # less frequent target updates = more stable

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(2)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state_t)).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Reward normalization — clip to [-10, 10] to tame scale mismatch
        rewards = torch.clamp(rewards, -10.0, 10.0)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q  = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q    = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
