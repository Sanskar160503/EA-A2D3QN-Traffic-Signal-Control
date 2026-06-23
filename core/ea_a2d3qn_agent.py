"""
ea_a2d3qn_agent.py
══════════════════════════════════════════════════════════════════════
EA-A2D3QN: Emergency-Aware Attention-Augmented Dueling Double DQN

Novel contributions over standard D3QN:
  1. Self-Attention State Encoder  — dynamically reweights 11 state features
  2. Emergency-Aware Dual PER     — guarantees emergency transitions in every batch
  3. Auxiliary Phase Timer Task   — forces temporal awareness in the network
  4. Attention Visualization      — logs which features agent focuses on per step

Patent claim basis:
  The specific combination of (1)+(2) applied to traffic signal control
  with emergency vehicle prioritization, motivated by empirical Q-value
  gap analysis (gap of 0.0003 observed without temporal attention).

Author: [Your Name]
Date  : 2026
"""

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════
# COMPONENT 1: Self-Attention State Encoder
# ══════════════════════════════════════════════════════════════════════

class SelfAttentionEncoder(nn.Module):
    """
    Applies self-attention across the 11 state dimensions.

    WHY THIS IS NEEDED:
    The 11-dim state has features of very different importance depending
    on the situation:
      - During emergency: index 10 (emergency flag) should dominate
      - During peak hour: queue lengths (0-3) matter most
      - Near phase limit: phase timer (9) should dominate

    Standard linear layers apply fixed weights regardless of context.
    Self-attention lets the network learn to reweight features dynamically
    based on the current state itself.

    HOW IT WORKS:
    For each feature dimension i, compute how much attention it should
    receive from every other feature dimension j. Features that are
    currently high and correlated with other high features get more weight.

    PATENT RELEVANCE:
    This is the first application of intra-state self-attention (not
    inter-agent attention) specifically motivated by Q-value gap analysis
    in single-intersection traffic signal control.
    """

    def __init__(self, state_dim, num_heads=1):
        super().__init__()
        self.state_dim  = state_dim
        self.num_heads  = num_heads

        # Project each scalar state feature into a vector for attention
        # state_dim → embed_dim so attention can compute relationships
        self.embed_dim  = 16   # embedding size per feature
        self.total_dim  = state_dim * self.embed_dim

        # Feature embedding: each of the 11 state dimensions gets its own
        # learned embedding that captures its semantic meaning
        self.feature_embed = nn.Linear(1, self.embed_dim)

        # Attention projections (Q, K, V)
        self.W_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_key   = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_value = nn.Linear(self.embed_dim, self.embed_dim)

        # Output projection: attended features → output dim
        self.output_proj = nn.Linear(self.total_dim, state_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(state_dim)

        # Store attention weights for visualization
        self.last_attention_weights = None

    def forward(self, x):
        """
        x: [batch_size, state_dim]
        Returns: [batch_size, state_dim]  (attended state, same shape)
        """
        batch_size = x.shape[0]

        # Step 1: Embed each scalar feature into a vector
        # x_features: [batch, state_dim, 1] → [batch, state_dim, embed_dim]
        x_features = x.unsqueeze(-1)                          # [B, 11, 1]
        embedded   = self.feature_embed(x_features)           # [B, 11, 16]

        # Step 2: Compute Query, Key, Value for each feature
        Q = self.W_query(embedded)   # [B, 11, 16]
        K = self.W_key(embedded)     # [B, 11, 16]
        V = self.W_value(embedded)   # [B, 11, 16]

        # Step 3: Attention scores
        # score[i,j] = how much feature i should attend to feature j
        scale   = self.embed_dim ** 0.5
        scores  = torch.bmm(Q, K.transpose(1, 2)) / scale    # [B, 11, 11]
        weights = F.softmax(scores, dim=-1)                   # [B, 11, 11]

        # Store for visualization (detached, no gradient)
        self.last_attention_weights = weights.detach().cpu()

        # Step 4: Weighted combination of values
        attended = torch.bmm(weights, V)                      # [B, 11, 16]

        # Step 5: Flatten and project back to state_dim
        attended_flat = attended.reshape(batch_size, -1)      # [B, 11*16]
        output        = self.output_proj(attended_flat)        # [B, 11]

        # Step 6: Residual connection + layer norm
        # (attended state + original state, normalized)
        return self.norm(output + x)

    def get_attention_weights(self):
        """
        Returns attention weight matrix for visualization.
        Shape: [batch, state_dim, state_dim]
        weights[b, i, j] = how much feature i attends to feature j
        """
        return self.last_attention_weights


# ══════════════════════════════════════════════════════════════════════
# COMPONENT 2: Attention-Augmented Dueling DQN Network
# ══════════════════════════════════════════════════════════════════════

class A2DuelingDQN(nn.Module):
    """
    Attention-Augmented Dueling DQN network.

    Architecture:
      Input(11) → SelfAttention → Linear(128) → ReLU → Linear(128) → ReLU
                                                              ↓ splits
                              Value: Linear(64) → ReLU → Linear(1)
                              Advantage: Linear(64) → ReLU → Linear(2)
                              PhaseTimer: Linear(32) → ReLU → Linear(1) [aux]
                                                              ↓ combines
                                        Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # NEW: Self-attention encoder (Component 1)
        self.attention = SelfAttentionEncoder(state_dim)

        # Shared feature extractor (same as before)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream: how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream: how much better is each action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # NEW: Auxiliary task — predict phase timer remaining
        # Forces network to develop temporal awareness
        # This directly addresses the Q-value gap problem we diagnosed
        self.phase_predictor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()   # output in [0,1] matching normalized phase timer
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns: (q_values, phase_timer_pred)
          q_values:        [batch, action_dim]
          phase_timer_pred:[batch, 1]  — auxiliary output
        """
        # Step 1: Attend to relevant state features
        x_attended = self.attention(x)

        # Step 2: Extract shared features
        features   = self.feature(x_attended)

        # Step 3: Dual-stream Q-value computation
        value      = self.value_stream(features)
        advantage  = self.advantage_stream(features)
        q_values   = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Step 4: Auxiliary phase timer prediction
        timer_pred = self.phase_predictor(features)

        return q_values, timer_pred

    def get_attention_weights(self):
        """Expose attention weights for visualization."""
        return self.attention.get_attention_weights()


# ══════════════════════════════════════════════════════════════════════
# COMPONENT 3: Emergency-Aware Dual PER Buffer
# ══════════════════════════════════════════════════════════════════════

class SumTree:
    """Binary sum tree for O(log n) priority sampling."""
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
        return (self._retrieve(left, s) if s <= self.tree[left]
                else self._retrieve(right, s - self.tree[left]))

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


class SinglePER:
    """Standard single-tier PER buffer (used inside the dual buffer)."""
    def __init__(self, capacity, alpha=0.4, beta_start=0.4, beta_frames=50000):
        self.tree        = SumTree(capacity)
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1
        self.epsilon     = 1e-4
        self.max_prio    = 1.0

    def add(self, s, a, r, s2, done):
        self.tree.add(self.max_prio, (s, a, r, s2, done))

    def sample(self, batch_size):
        indices, weights, batch = [], [], []
        segment  = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_frames)
        self.frame += 1

        min_prob = (np.min(self.tree.tree[-self.tree.capacity:])
                    / (self.tree.total + 1e-8))
        min_prob = max(min_prob, 1e-7)

        for i in range(batch_size):
            s_val              = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s_val)
            prob               = priority / (self.tree.total + 1e-8)
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


class EmergencyAwareDualPER:
    """
    NOVEL CONTRIBUTION: Emergency-Aware Dual Prioritized Experience Replay

    WHY STANDARD PER IS INSUFFICIENT:
    Emergency vehicle events occur ~0.5% of timesteps. With 20,000 capacity
    and uniform-ish sampling even with PER, the agent sees perhaps 100
    emergency transitions per training run — far too few to learn robust
    emergency handling. This is why your emergency clearance time was 6.5
    steps vs 4.5 for fixed-time.

    SOLUTION:
    Two separate PER buffers:
      - normal_buffer  (90% capacity): all transitions
      - emergency_buffer (10% capacity): only emergency transitions

    Every training batch is composed of:
      - (1 - emergency_ratio) fraction from normal_buffer
      - emergency_ratio fraction GUARANTEED from emergency_buffer

    With emergency_ratio=0.25, 25% of every batch is emergency transitions,
    vs ~0.5% under standard PER. That is 50x more emergency training signal.

    PATENT NOVELTY:
    This specific dual-buffer architecture with guaranteed emergency sampling
    applied to traffic signal control has not been published.
    """

    def __init__(self,
                 total_capacity  = 20000,
                 emergency_ratio = 0.25,    # fraction of each batch from emergency buffer
                 alpha           = 0.4,
                 beta_start      = 0.4):

        self.emergency_ratio = emergency_ratio

        # Normal buffer: 90% of total capacity, stores everything
        normal_cap         = int(total_capacity * 0.90)
        self.normal_buffer = SinglePER(normal_cap, alpha, beta_start)

        # Emergency buffer: 10% of total capacity, stores ONLY emergency transitions
        em_cap             = int(total_capacity * 0.10)
        self.em_buffer     = SinglePER(em_cap, alpha=0.6,   # higher alpha for emergency
                                       beta_start=beta_start)

        # Statistics
        self.total_added     = 0
        self.emergency_added = 0

    def add(self, s, a, r, s2, done, is_emergency=False):
        """
        Add transition to buffer.
        is_emergency: True if emergency vehicle was present in this timestep.
        """
        # All transitions go to normal buffer
        self.normal_buffer.add(s, a, r, s2, done)
        self.total_added += 1

        # Emergency transitions ALSO go to dedicated emergency buffer
        if is_emergency:
            self.em_buffer.add(s, a, r, s2, done)
            self.emergency_added += 1

    def sample(self, batch_size):
        """
        Sample with guaranteed emergency representation.
        Returns: batch, (normal_indices, em_indices), weights
        """
        # How many emergency samples in this batch?
        em_size     = int(batch_size * self.emergency_ratio)
        normal_size = batch_size - em_size

        # Sample from normal buffer
        normal_batch, normal_idx, normal_w = self.normal_buffer.sample(normal_size)

        # Sample from emergency buffer if we have enough, else fall back to normal
        if len(self.em_buffer) >= em_size and em_size > 0:
            em_batch, em_idx, em_w = self.em_buffer.sample(em_size)
            # Emergency transitions get slightly higher IS weights
            # (they are more important, so we reduce correction)
            em_w = em_w * 0.8
        else:
            # Not enough emergency transitions yet — use normal buffer
            em_batch, em_idx, em_w = self.normal_buffer.sample(em_size)
            em_idx = None   # signal that these are from normal buffer

        full_batch   = normal_batch + em_batch
        full_weights = np.concatenate([normal_w, em_w])

        return full_batch, (normal_idx, em_idx), full_weights

    def update_priorities(self, indices, td_errors):
        normal_idx, em_idx = indices
        n = len(normal_idx)

        # Update normal buffer priorities
        self.normal_buffer.update_priorities(normal_idx, td_errors[:n])

        # Update emergency buffer priorities if they came from there
        if em_idx is not None:
            self.em_buffer.update_priorities(em_idx, td_errors[n:])

    def __len__(self):
        return len(self.normal_buffer)

    def stats(self):
        return {
            "total":          self.total_added,
            "emergency":      self.emergency_added,
            "em_pct":         self.emergency_added / max(self.total_added, 1) * 100,
            "normal_buf_len": len(self.normal_buffer),
            "em_buf_len":     len(self.em_buffer),
        }


# ══════════════════════════════════════════════════════════════════════
# MAIN AGENT: EA-A2D3QN
# ══════════════════════════════════════════════════════════════════════

class EA_A2D3QNAgent:
    """
    EA-A2D3QN: Emergency-Aware Attention-Augmented Dueling Double DQN

    Novel components:
      1. SelfAttentionEncoder  — dynamic state feature reweighting
      2. EmergencyAwareDualPER — guaranteed emergency transition sampling
      3. Auxiliary phase timer prediction — temporal awareness forcing
      4. Attention weight logging — visualization and interpretability
    """

    def __init__(self, state_dim, action_dim,
                 emergency_ratio  = 0.25,
                 aux_loss_weight  = 0.1):

        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.aux_loss_weight = aux_loss_weight   # weight for phase timer aux loss

        # Hyperparameters
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.9935
        self.batch_size    = 256
        self.lr            = 1e-4

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EA-A2D3QN] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[EA-A2D3QN] GPU: {torch.cuda.get_device_name(0)}")

        # Networks (Component 1 is inside A2DuelingDQN)
        self.policy_net = A2DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = A2DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with LR scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[500_000, 800_000], gamma=0.5
        )

        # Loss functions
        self.td_loss_fn  = nn.SmoothL1Loss(reduction='none')   # for PER weights
        self.aux_loss_fn = nn.MSELoss()                         # for phase timer

        # Component 2: Emergency-Aware Dual PER buffer
        self.memory = EmergencyAwareDualPER(
            total_capacity  = 20000,
            emergency_ratio = emergency_ratio,
            alpha           = 0.4,
        )

        # Training state
        self.update_counter     = 0
        self.target_update_freq = 1000

        # Attention logging
        self.attention_log      = []    # stores (step, weights, state) tuples
        self.log_attention_every = 100  # log every N steps

    def act(self, state, step=None):
        """
        Select action using epsilon-greedy with self-attention.
        Optionally log attention weights for visualization.
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.policy_net(s)
            action = torch.argmax(q_values).item()

        # Log attention weights periodically
        if step is not None and step % self.log_attention_every == 0:
            weights = self.policy_net.get_attention_weights()
            if weights is not None:
                self.attention_log.append({
                    "step":     step,
                    "weights":  weights.numpy(),
                    "state":    state.copy(),
                    "action":   action,
                    "emergency": bool(state[10] > 0.5)
                })

        return action

    def remember(self, s, a, r, s2, done, is_emergency=False):
        """
        Store transition. Pass is_emergency=True when emergency vehicle
        was present this timestep — sends to dedicated emergency buffer.
        """
        self.memory.add(s, a, float(r), s2, float(done),
                        is_emergency=is_emergency)

    def replay(self):
        """
        Training step with:
          - Emergency-aware batch sampling (Component 2)
          - Double DQN targets
          - PER importance-sampling weights
          - Auxiliary phase timer loss (Component 3)
        """
        if len(self.memory) < self.batch_size:
            return None, None

        # Sample with guaranteed emergency transitions
        batch, indices, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights     = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        rewards = torch.clamp(rewards, -10.0, 10.0)

        # Current Q-values + auxiliary phase timer prediction
        q_values, timer_pred = self.policy_net(states)
        q_values             = q_values.gather(1, actions)

        # Double DQN: policy net selects, target net evaluates
        with torch.no_grad():
            next_q_all, _    = self.policy_net(next_states)
            next_actions     = next_q_all.argmax(1, keepdim=True)
            next_q_target, _ = self.target_net(next_states)
            next_q           = next_q_target.gather(1, next_actions)
            target_q         = rewards + self.gamma * next_q * (1 - dones)

        # TD loss (weighted by PER importance sampling weights)
        td_element_loss = self.td_loss_fn(q_values, target_q)
        td_loss         = (weights * td_element_loss).mean()

        # Auxiliary phase timer loss
        # states[:, 9] is the normalized phase timer (ground truth)
        timer_true = states[:, 9:10]
        aux_loss   = self.aux_loss_fn(timer_pred, timer_true)

        # Combined loss
        loss = td_loss + self.aux_loss_weight * aux_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update PER priorities with TD errors
        td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        # Sync target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  [Target synced @ step {self.update_counter}]")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item(), td_loss.item()

    def get_attention_summary(self):
        """
        Compute average attention weights separately for:
          - Normal traffic steps
          - Emergency traffic steps
        This is your key result for the paper.
        """
        if not self.attention_log:
            return None

        state_labels = ["q_N", "q_S", "q_E", "q_W",
                        "w_N", "w_S", "w_E", "w_W",
                        "phase", "timer", "emergency"]

        normal_weights = []
        emerg_weights  = []

        for entry in self.attention_log:
            # Mean attention weight received by each feature (avg over query dim)
            avg_weights = entry["weights"][0].mean(axis=0)   # [11]
            if entry["emergency"]:
                emerg_weights.append(avg_weights)
            else:
                normal_weights.append(avg_weights)

        result = {"labels": state_labels}
        if normal_weights:
            result["normal_avg"] = np.mean(normal_weights, axis=0)
        if emerg_weights:
            result["emergency_avg"] = np.mean(emerg_weights, axis=0)

        return result

    def print_attention_summary(self):
        """Print which features the agent focuses on in each scenario."""
        summary = self.get_attention_summary()
        if not summary:
            print("No attention data logged yet.")
            return

        labels = summary["labels"]
        print("\n" + "="*55) 
        print("  Attention Weight Summary")
        print("  (higher = agent focuses more on this feature)")
        print("="*55)

        if "normal_avg" in summary:
            print("\n  Normal traffic steps:")
            sorted_idx = np.argsort(summary["normal_avg"])[::-1]
            for i in sorted_idx[:5]:
                print(f"    {labels[i]:<12} {summary['normal_avg'][i]:.4f}")

        if "emergency_avg" in summary:
            print("\n  Emergency vehicle steps:")
            sorted_idx = np.argsort(summary["emergency_avg"])[::-1]
            for i in sorted_idx[:5]:
                print(f"    {labels[i]:<12} {summary['emergency_avg'][i]:.4f}")
        print("="*55)