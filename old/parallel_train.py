# parallel_train.py
# Runs N SUMO instances in parallel processes, each collecting experience
# into a shared queue. The main process handles GPU replay only.
# This keeps the GPU busy instead of waiting on SUMO.

import torch
import numpy as np
import multiprocessing as mp
from collections import deque
import random

from agent_v2 import DQNAgent
from environment import TrafficEnv

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

NUM_WORKERS  = 3       # number of parallel SUMO instances
REPLAY_SIZE  = 30000
BATCH_SIZE   = 256
EPISODES     = 1000
MAX_STEPS    = 1000


def worker_fn(worker_id, experience_queue, param_queue, epsilon_val):
    """Each worker runs its own SUMO instance and pushes (s,a,r,s',done) tuples."""
    env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)

    # Local lightweight policy (CPU only, no optimizer)
    import torch.nn as nn
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        def forward(self, x):
            return self.net(x)

    policy = Policy()

    for episode in range(EPISODES // NUM_WORKERS):
        # Sync weights from main process if available
        if not param_queue.empty():
            state_dict = param_queue.get()
            policy.load_state_dict(state_dict)

        state = env.reset(use_gui=False)
        epsilon = epsilon_val.value

        for step in range(MAX_STEPS):
            if random.random() < epsilon:
                action = random.randrange(2)
            else:
                with torch.no_grad():
                    t = torch.FloatTensor(state).unsqueeze(0)
                    action = torch.argmax(policy(t)).item()

            next_state, reward, done = env.step(action)
            reward_norm = np.clip(reward / 100.0, -10.0, 10.0)
            experience_queue.put((state, action, reward_norm, next_state, done))
            state = next_state
            if done:
                break

    env.close()


def train_parallel():
    mp.set_start_method('spawn', force=True)

    experience_queue = mp.Queue(maxsize=5000)
    param_queue      = mp.Queue(maxsize=NUM_WORKERS)
    epsilon_val      = mp.Value('f', 1.0)

    # Start worker processes
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_fn, args=(i, experience_queue, param_queue, epsilon_val))
        p.start()
        workers.append(p)

    # Main process: GPU replay
    agent  = DQNAgent(state_dim=6, action_dim=2)
    memory = deque(maxlen=REPLAY_SIZE)
    step   = 0

    print(f"Training with {NUM_WORKERS} parallel SUMO workers...")
    print(f"Device: {agent.device}")

    while any(p.is_alive() for p in workers):
        # Drain experience queue into replay buffer
        drained = 0
        while not experience_queue.empty() and drained < 500:
            memory.append(experience_queue.get_nowait())
            drained += 1

        # GPU replay — only runs when buffer is full enough
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            device = agent.device
            states_t      = torch.FloatTensor(np.array(states)).to(device)
            actions_t     = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
            dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_vals = agent.policy_net(states_t).gather(1, actions_t)
            with torch.no_grad():
                max_next = agent.target_net(next_states_t).max(1)[0].unsqueeze(1)
                target   = rewards_t + 0.95 * max_next * (1 - dones_t)

            loss = agent.loss_fn(q_vals, target)
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
            agent.optimizer.step()

            step += 1

            # Sync weights back to workers every 200 updates
            if step % 200 == 0:
                cpu_weights = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
                for _ in range(NUM_WORKERS):
                    try:
                        param_queue.put_nowait(cpu_weights)
                    except:
                        pass

            # Decay epsilon
            if step % 50 == 0:
                epsilon_val.value = max(0.05, epsilon_val.value * 0.998)

            if step % 500 == 0:
                print(f"Step {step:6d} | Buffer: {len(memory):5d} | "
                      f"Loss: {loss.item():.4f} | Epsilon: {epsilon_val.value:.3f}")

    for p in workers:
        p.join()

    torch.save(agent.policy_net.state_dict(), "dqn_parallel.pth")
    print("Done. Model saved → dqn_parallel.pth")


if __name__ == "__main__":
    train_parallel()
