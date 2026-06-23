#train.py
import numpy as np
import matplotlib.pyplot as plt
import torch

from agent import DQNAgent
from environment import TrafficEnv

# -------- SUMO Paths ----------
SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG = "traffic.sumocfg"

# -------- Training Settings ----------
EPISODES = 5000
MAX_STEPS = 1000

# -------- Initialize Environment ----------
env = TrafficEnv(
    sumo_binary=SUMO_BINARY,
    sumo_gui_binary=SUMO_GUI_BINARY,
    sumo_cfg=SUMO_CFG,
    sumo_workdir=SUMO_WORKDIR
)

# State = 4 queue lanes + traffic light phase  → 5
state_dim = 6
action_dim = 2  # 0 = keep, 1 = switch

agent = DQNAgent(state_dim, action_dim)
reward_history = []

# -------- Training Loop ----------
for episode in range(EPISODES):
    state = env.reset(use_gui=False)   # GUI OFF for training
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            break
    
    reward_history.append(total_reward)
    print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

env.close()

# -------- Save Model ----------
MODEL_PATH = "dqn_model.pth"
torch.save(agent.policy_net.state_dict(), MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

# -------- Plot Learning Curve ----------
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance")
plt.grid()
plt.show()

# -------- Save Reward Data ----------
np.save("reward_history.npy", reward_history)
print("Training completed successfully.")