import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from agent_v3 import D3QNAgent
from environment import TrafficEnv

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

# ─────────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────────
EPISODES      = 1000
MAX_STEPS     = 1000
WARMUP_EP     = 100      # reduced from 200 — model converges faster now
LOG_INTERVAL  = 50
MODEL_PATH    = "dqn_d3qn.pth"

# Set RESUME_FROM to a .pth path to continue training from checkpoint
# Set to None to train from scratch
RESUME_FROM   = None     # e.g. "dqn_d3qn.pth"

# How many replay() calls per environment step
# Increasing this makes better use of GPU between SUMO steps
REPLAY_PER_STEP = 2

# ─────────────────────────────────────────────
#  Init
# ─────────────────────────────────────────────
env = TrafficEnv(
    sumo_binary=SUMO_BINARY,
    sumo_gui_binary=SUMO_GUI_BINARY,
    sumo_cfg=SUMO_CFG,
    sumo_workdir=SUMO_WORKDIR
)

state_dim  = 6
action_dim = 2
agent      = D3QNAgent(state_dim, action_dim)

if RESUME_FROM:
    agent.policy_net.load_state_dict(torch.load(RESUME_FROM, map_location=agent.device))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.epsilon = 0.2    # resume with low epsilon — already somewhat trained
    print(f"Resumed from {RESUME_FROM} | epsilon set to {agent.epsilon}")

reward_history    = []
loss_history      = []
queue_history     = []
best_avg_reward   = -np.inf
prev_queue        = None     # for reward shaping (queue improvement bonus)
train_start       = time.time()

print(f"\n{'='*65}")
print(f"  D3QN Training  |  {EPISODES} eps × {MAX_STEPS} steps  |  {REPLAY_PER_STEP}x replay/step")
print(f"  epsilon decay: 0.9935  →  ~0.05 by ep 700")
print(f"  LR: 1e-4  →  5e-5 at ep 500  →  2.5e-5 at ep 800")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────
for episode in range(EPISODES):
    state        = env.reset(use_gui=False)
    total_reward = 0.0
    ep_losses    = []
    ep_queues    = []
    prev_queue   = float(np.sum(state[:4]))

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # ── Reward shaping: bonus if queue improved this step ──
        curr_queue = float(np.sum(next_state[:4]))
        queue_delta = prev_queue - curr_queue     # positive = queue got shorter
        shaped_reward = reward + 0.5 * queue_delta
        prev_queue = curr_queue

        reward_norm = float(np.clip(shaped_reward / 100.0, -10.0, 10.0))

        agent.remember(state, action, reward_norm, next_state, float(done))

        # Multiple replay passes per step — better GPU utilization
        for _ in range(REPLAY_PER_STEP):
            loss = agent.replay()
            if loss is not None:
                ep_losses.append(loss)

        ep_queues.append(curr_queue)
        state        = next_state
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)
    loss_history.append(np.mean(ep_losses) if ep_losses else 0.0)
    queue_history.append(np.mean(ep_queues))

    if episode >= WARMUP_EP:
        avg50 = np.mean(reward_history[-50:])
        if avg50 > best_avg_reward:
            best_avg_reward = avg50
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    if (episode + 1) % LOG_INTERVAL == 0:
        avg50     = np.mean(reward_history[-50:])
        avg_queue = np.mean(queue_history[-50:])
        avg_loss  = np.mean(loss_history[-50:])
        elapsed   = (time.time() - train_start) / 60
        remaining = elapsed / (episode + 1) * (EPISODES - episode - 1)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(
            f"Ep {episode+1:4d}/{EPISODES} | "
            f"Avg50 R: {avg50:7.1f} | "
            f"Queue: {avg_queue:5.1f} | "
            f"Loss: {avg_loss:.5f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"LR: {current_lr:.1e} | "
            f"Buf: {len(agent.memory):5d} | "
            f"{elapsed:.0f}m/{remaining:.0f}m"
        )

# ─────────────────────────────────────────────
#  Save & Plot
# ─────────────────────────────────────────────
env.close()
total_time = (time.time() - train_start) / 60
print(f"\nDone in {total_time:.1f} min | Best model → {MODEL_PATH} (avg50: {best_avg_reward:.1f})")

np.save("reward_history_d3qn.npy", reward_history)
np.save("loss_history_d3qn.npy",   loss_history)
np.save("queue_history_d3qn.npy",  queue_history)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("D3QN Training Dashboard (v2 — fixed)", fontsize=14, fontweight='bold')
window = 50
eps    = range(1, EPISODES + 1)

ax = axes[0, 0]
ma = np.convolve(reward_history, np.ones(window) / window, mode='valid')
ax.plot(eps, reward_history, alpha=0.2, color='steelblue')
ax.plot(range(window, EPISODES + 1), ma, color='steelblue', linewidth=2, label=f'{window}-ep avg')
ax.axhline(best_avg_reward, color='green', linestyle='--', linewidth=1, label=f'Best avg: {best_avg_reward:.0f}')
ax.set_title("Total reward per episode")
ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
loss_ma = np.convolve(loss_history, np.ones(window) / window, mode='valid')
ax.plot(eps, loss_history, alpha=0.2, color='coral')
ax.plot(range(window, EPISODES + 1), loss_ma, color='coral', linewidth=2)
ax.set_title("Huber loss per episode")
ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
ax.grid(alpha=0.3)

ax = axes[1, 0]
q_ma = np.convolve(queue_history, np.ones(window) / window, mode='valid')
ax.plot(eps, queue_history, alpha=0.2, color='mediumpurple')
ax.plot(range(window, EPISODES + 1), q_ma, color='mediumpurple', linewidth=2)
ax.set_title("Avg queue length per episode")
ax.set_xlabel("Episode"); ax.set_ylabel("Vehicles waiting")
ax.grid(alpha=0.3)

ax = axes[1, 1]
# Actual epsilon values from training
ep_vals = [max(0.05, 1.0 * (0.9935 ** i)) for i in range(EPISODES)]
ax.plot(eps, ep_vals, color='darkorange', linewidth=2)
ax.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='min ε = 0.05')
ax.set_title("Epsilon decay")
ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_d3qn_v2.png", dpi=150)
plt.show()
print("Plot saved → training_d3qn_v2.png")
