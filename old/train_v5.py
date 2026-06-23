# train_v3.py — Final version with rebalanced reward
# state_dim = 11, reward target range [-300, +300] per episode

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from collections import Counter

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
MODEL_PATH      = "dqn_final.pth"
RESUME_FROM     = None
REPLAY_PER_STEP = 2

state_dim  = 11
action_dim = 2

env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
agent = D3QNAgent(state_dim, action_dim)

if RESUME_FROM:
    try:
        agent.policy_net.load_state_dict(
            torch.load(RESUME_FROM, map_location=agent.device, weights_only=True)
        )
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.epsilon = 0.3
        print(f"Resumed from {RESUME_FROM}")
    except Exception as e:
        print(f"Could not load checkpoint: {e} — training from scratch")

reward_history  = []
loss_history    = []
queue_history   = []
phase_history   = []
best_avg_reward = -np.inf
train_start     = time.time()

print(f"\n{'='*70}")
print(f"  D3QN Final  |  state_dim=11  |  {EPISODES} eps")
print(f"  Reward range target: [-300, +300] per episode")
print(f"  Reward normalization: /1.0 (already scaled in env)")
print(f"{'='*70}\n")

for episode in range(EPISODES):
    state        = env.reset(use_gui=False)
    total_reward = 0.0
    ep_losses    = []
    ep_queues    = []
    ep_phases    = []

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # Reward already in small range — just clamp, don't divide
        reward_norm = float(np.clip(reward, -2.0, 2.0))

        agent.remember(state, action, reward_norm, next_state, float(done))

        for _ in range(REPLAY_PER_STEP):
            loss = agent.replay()
            if loss is not None:
                ep_losses.append(loss)

        ep_queues.append(float(np.sum(next_state[:4]) * 10.0))  # denormalize for display
        ep_phases.append(int(round(next_state[8])))

        state        = next_state
        total_reward += reward
        if done:
            break

    reward_history.append(total_reward)
    loss_history.append(np.mean(ep_losses) if ep_losses else 0.0)
    queue_history.append(np.mean(ep_queues))

    phase_counts = Counter(ep_phases)
    total_steps  = max(len(ep_phases), 1)
    phase_fracs  = [phase_counts.get(i, 0) / total_steps for i in range(4)]
    phase_history.append(np.std(phase_fracs))

    if episode >= WARMUP_EP:
        avg50 = np.mean(reward_history[-50:])
        if avg50 > best_avg_reward:
            best_avg_reward = avg50
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    if (episode + 1) % LOG_INTERVAL == 0:
        avg50      = np.mean(reward_history[-50:])
        avg_queue  = np.mean(queue_history[-50:])
        avg_loss   = np.mean(loss_history[-50:])
        avg_pbias  = np.mean(phase_history[-50:])
        elapsed    = (time.time() - train_start) / 60
        remaining  = elapsed / (episode + 1) * (EPISODES - episode - 1)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(
            f"Ep {episode+1:4d}/{EPISODES} | "
            f"R: {avg50:7.1f} | "
            f"Q: {avg_queue:4.1f} | "
            f"PhBias: {avg_pbias:.3f} | "
            f"Loss: {avg_loss:.5f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"LR: {current_lr:.1e} | "
            f"{elapsed:.0f}m/{remaining:.0f}m"
        )

env.close()
total_time = (time.time() - train_start) / 60
print(f"\nDone in {total_time:.1f} min | Best → {MODEL_PATH} (avg50: {best_avg_reward:.1f})")

np.save("reward_history_final.npy", reward_history)
np.save("queue_history_final.npy",  queue_history)
np.save("loss_history_final.npy",   loss_history)
np.save("phase_history_final.npy",  phase_history)

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("D3QN Training Dashboard (Final)", fontsize=14, fontweight='bold')
window = 50
eps    = range(1, EPISODES + 1)

ax = axes[0, 0]
ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
ax.plot(eps, reward_history, alpha=0.2, color='steelblue')
ax.plot(range(window, EPISODES+1), ma, color='steelblue', linewidth=2, label=f'{window}-ep avg')
ax.axhline(best_avg_reward, color='green', linestyle='--', linewidth=1, label=f'Best: {best_avg_reward:.1f}')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.set_title("Total reward per episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
loss_ma = np.convolve(loss_history, np.ones(window)/window, mode='valid')
ax.plot(eps, loss_history, alpha=0.2, color='coral')
ax.plot(range(window, EPISODES+1), loss_ma, color='coral', linewidth=2)
ax.set_title("Huber loss per episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

ax = axes[1, 0]
q_ma = np.convolve(queue_history, np.ones(window)/window, mode='valid')
ax.plot(eps, queue_history, alpha=0.2, color='mediumpurple')
ax.plot(range(window, EPISODES+1), q_ma, color='mediumpurple', linewidth=2)
ax.set_title("Avg queue length per episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Vehicles waiting")
ax.grid(alpha=0.3)

ax = axes[1, 1]
p_ma = np.convolve(phase_history, np.ones(window)/window, mode='valid')
ax.plot(eps, phase_history, alpha=0.2, color='darkorange')
ax.plot(range(window, EPISODES+1), p_ma, color='darkorange', linewidth=2)
ax.axhline(0.0, color='green', linestyle='--', linewidth=1, label='Perfect balance')
ax.set_title("Phase imbalance (↓ better)"); ax.set_xlabel("Episode")
ax.set_ylabel("Std of phase fractions")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_final.png", dpi=150)
plt.show()
print("Plot saved → training_final.png")
