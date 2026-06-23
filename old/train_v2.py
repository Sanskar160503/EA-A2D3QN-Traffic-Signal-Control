# train_v2.py — Stable training with logging and moving average tracking
import numpy as np
import matplotlib.pyplot as plt
import torch

from agent_v2 import DQNAgent
from environment import TrafficEnv   # your existing environment.py

# -------- SUMO Paths ---------- 
SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

EPISODES  = 1000
MAX_STEPS = 1000
WARMUP    = 200   # episodes before we start tracking convergence

env = TrafficEnv(
    sumo_binary=SUMO_BINARY,
    sumo_gui_binary=SUMO_GUI_BINARY,
    sumo_cfg=SUMO_CFG,
    sumo_workdir=SUMO_WORKDIR
)

state_dim  = 6
action_dim = 2
agent      = DQNAgent(state_dim, action_dim)

reward_history  = []
loss_history    = []
best_avg_reward = -np.inf
MODEL_PATH      = "dqn_model_v2.pth"

for episode in range(EPISODES):
    state      = env.reset(use_gui=False)
    total_reward = 0
    ep_losses  = []

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # Normalize reward here if you haven't changed environment.py yet
        reward_norm = np.clip(reward / 100.0, -10.0, 10.0)

        agent.remember(state, action, reward_norm, next_state, done)
        loss = agent.replay()
        if loss is not None:
            ep_losses.append(loss)

        state = next_state
        total_reward += reward   # log raw reward for interpretability

        if done:
            break

    reward_history.append(total_reward)
    avg_loss = np.mean(ep_losses) if ep_losses else 0.0
    loss_history.append(avg_loss)

    # Save best model based on 50-episode moving average
    if episode >= WARMUP:
        avg50 = np.mean(reward_history[-50:])
        if avg50 > best_avg_reward:
            best_avg_reward = avg50
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

    if (episode + 1) % 50 == 0:
        avg50 = np.mean(reward_history[-50:])
        print(
            f"Ep {episode+1:4d} | "
            f"Avg50 Reward: {avg50:8.1f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

env.close()
print(f"\nBest model saved → {MODEL_PATH}  (avg50 reward: {best_avg_reward:.1f})")

# -------- Plots ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

window = 50
moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')

ax1.plot(reward_history, alpha=0.3, color='steelblue', label='Raw reward')
ax1.plot(range(window-1, len(reward_history)), moving_avg,
         color='steelblue', linewidth=2, label=f'{window}-ep moving avg')
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title("DQN Training Performance (v2)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(loss_history, alpha=0.5, color='coral', label='Loss per episode')
ax2.set_xlabel("Episode")
ax2.set_ylabel("Huber Loss")
ax2.set_title("Training Loss")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_v2.png", dpi=150)
plt.show()

np.save("reward_history_v2.npy", reward_history)
print("Done.")
