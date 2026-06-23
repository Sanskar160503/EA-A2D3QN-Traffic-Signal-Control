# state: [q_N, q_S, q_E, q_W, wait_N, wait_S, wait_E, wait_W, phase, phase_timer_norm, emergency]

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

EPISODES        = 1000
MAX_STEPS       = 1000
WARMUP_EP       = 100
LOG_INTERVAL    = 50
MODEL_PATH      = "dqn_d3qn_v3.pth"
RESUME_FROM     = None
REPLAY_PER_STEP = 2

# ── 11-dimensional state ──
state_dim  = 11
action_dim = 2

env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
agent = D3QNAgent(state_dim, action_dim)

if RESUME_FROM:
    try:
        agent.policy_net.load_state_dict(torch.load(RESUME_FROM, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.epsilon = 0.3
        print(f"Resumed from {RESUME_FROM}")
    except Exception as e:
        print(f"Could not load {RESUME_FROM}: {e} — training from scratch")

reward_history  = []
loss_history    = []
queue_history   = []
wait_history    = []
phase_history   = []    # track phase distribution per episode
best_avg_reward = -np.inf
train_start     = time.time()

print(f"\n{'='*65}")
print(f"  D3QN v3  |  state_dim=11  |  {EPISODES} eps × {MAX_STEPS} steps")
print(f"  Key fix: phase_timer in state + forced phase rotation at 60 steps")
print(f"{'='*65}\n")

for episode in range(EPISODES):
    state        = env.reset(use_gui=False)
    total_reward = 0.0
    ep_losses    = []
    ep_queues    = []
    ep_waits     = []
    ep_phases    = []

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        reward_norm = float(np.clip(reward / 10.0, -10.0, 10.0))

        agent.remember(state, action, reward_norm, next_state, float(done))

        for _ in range(REPLAY_PER_STEP):
            loss = agent.replay()
            if loss is not None:
                ep_losses.append(loss)

        ep_queues.append(float(np.sum(next_state[:4])))
        ep_waits.append(float(np.sum(next_state[4:8])))
        ep_phases.append(int(next_state[8]))

        state        = next_state
        total_reward += reward
        if done:
            break

    reward_history.append(total_reward)
    loss_history.append(np.mean(ep_losses) if ep_losses else 0.0)
    queue_history.append(np.mean(ep_queues))
    wait_history.append(np.mean(ep_waits))

    # Track phase balance (std of phase counts — lower = more balanced)
    from collections import Counter
    phase_counts = Counter(ep_phases)
    total_steps  = len(ep_phases)
    phase_fracs  = [phase_counts.get(i, 0) / total_steps for i in range(4)]
    phase_history.append(np.std(phase_fracs))   # 0 = perfectly balanced

    if episode >= WARMUP_EP:
        avg50 = np.mean(reward_history[-50:])
        if avg50 > best_avg_reward:
            best_avg_reward = avg50
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    if (episode + 1) % LOG_INTERVAL == 0:
        avg50      = np.mean(reward_history[-50:])
        avg_queue  = np.mean(queue_history[-50:])
        avg_wait   = np.mean(wait_history[-50:])
        avg_loss   = np.mean(loss_history[-50:])
        avg_pbias  = np.mean(phase_history[-50:])
        elapsed    = (time.time() - train_start) / 60
        remaining  = elapsed / (episode + 1) * (EPISODES - episode - 1)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(
            f"Ep {episode+1:4d}/{EPISODES} | "
            f"R: {avg50:7.1f} | "
            f"Q: {avg_queue:4.1f} | "
            f"W: {avg_wait:.3f} | "
            f"PhBias: {avg_pbias:.3f} | "   # should decrease toward 0 = balanced phases
            f"Loss: {avg_loss:.5f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"LR: {current_lr:.1e} | "
            f"{elapsed:.0f}m/{remaining:.0f}m"
        )

env.close()
total_time = (time.time() - train_start) / 60
print(f"\nDone in {total_time:.1f} min | Best → {MODEL_PATH} (avg50: {best_avg_reward:.1f})")

np.save("reward_history_d3qn_v3.npy", reward_history)
np.save("queue_history_d3qn_v3.npy",  queue_history)
np.save("wait_history_d3qn_v3.npy",   wait_history)
np.save("loss_history_d3qn_v3.npy",   loss_history)
np.save("phase_history_d3qn_v3.npy",  phase_history)

# ── 4-panel plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("D3QN Training Dashboard (v3 — phase bias fixed)", fontsize=14, fontweight='bold')
window = 50
eps    = range(1, EPISODES + 1)

ax = axes[0, 0]
ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
ax.plot(eps, reward_history, alpha=0.2, color='steelblue')
ax.plot(range(window, EPISODES+1), ma, color='steelblue', linewidth=2, label=f'{window}-ep avg')
ax.axhline(best_avg_reward, color='green', linestyle='--', linewidth=1, label=f'Best: {best_avg_reward:.0f}')
ax.set_title("Total reward per episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
loss_ma = np.convolve(loss_history, np.ones(window)/window, mode='valid')
ax.plot(eps, loss_history, alpha=0.2, color='coral')
ax.plot(range(window, EPISODES+1), loss_ma, color='coral', linewidth=2)
ax.set_title("Huber loss per episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
ax.grid(alpha=0.3)

ax = axes[1, 0]
q_ma = np.convolve(queue_history, np.ones(window)/window, mode='valid')
ax.plot(range(window, EPISODES+1), q_ma, color='mediumpurple', linewidth=2, label='Avg queue')
ax2 = ax.twinx()
w_ma = np.convolve(wait_history, np.ones(window)/window, mode='valid')
ax2.plot(range(window, EPISODES+1), w_ma, color='teal', linewidth=1.5, linestyle='--', label='Avg wait (norm)')
ax.set_title("Queue & wait time"); ax.set_xlabel("Episode")
ax.set_ylabel("Vehicles waiting", color='mediumpurple')
ax2.set_ylabel("Normalized wait", color='teal')
ax.grid(alpha=0.3)

ax = axes[1, 1]
p_ma = np.convolve(phase_history, np.ones(window)/window, mode='valid')
ax.plot(eps, phase_history, alpha=0.2, color='darkorange')
ax.plot(range(window, EPISODES+1), p_ma, color='darkorange', linewidth=2)
ax.axhline(0.0, color='green', linestyle='--', linewidth=1, label='Perfect balance')
ax.set_title("Phase imbalance (↓ = more balanced)"); ax.set_xlabel("Episode"); ax.set_ylabel("Std of phase fractions")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_d3qn_v3.png", dpi=150)
plt.show()
print("Plot saved → training_d3qn_v3.png")
