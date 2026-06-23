# diagnose.py — Run this BEFORE retraining to understand the current policy
# Prints action distribution, Q-values, and phase switching behavior
# This will tell us WHY the queue is stuck at 6.0

import torch
import numpy as np
import traci
from agent_v3 import D3QNAgent
from environment import TrafficEnv
from collections import Counter

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MODEL_PATH      = "dqn_d3qn.pth"

env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
agent = D3QNAgent(state_dim=6, action_dim=2)
agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
agent.policy_net.eval()
agent.epsilon = 0.0   # pure greedy

state       = env.reset(use_gui=False)
actions     = []
q_vals_log  = []
phases      = []
queues      = []
switches    = 0
prev_action = None

for step in range(500):
    s_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_out = agent.policy_net(s_t)
    q_vals_log.append(q_out.cpu().numpy().flatten())

    action = torch.argmax(q_out).item()
    actions.append(action)
    phases.append(int(state[4]))
    queues.append(float(np.sum(state[:4])))

    if prev_action is not None and action != prev_action:
        switches += 1
    prev_action = action

    state, _, done = env.step(action)
    if done:
        break

env.close()

q_arr = np.array(q_vals_log)
print("\n" + "="*55)
print("  POLICY DIAGNOSIS")
print("="*55)

action_counts = Counter(actions)
print(f"\nAction distribution (500 steps):")
print(f"  KEEP  (0): {action_counts[0]:4d} steps  ({action_counts[0]/5:.1f}%)")
print(f"  SWITCH(1): {action_counts[1]:4d} steps  ({action_counts[1]/5:.1f}%)")
print(f"  Phase switches actually executed: {switches}")

print(f"\nQ-value statistics:")
print(f"  Q(s, KEEP)   mean={q_arr[:,0].mean():.4f}  std={q_arr[:,0].std():.4f}  range=[{q_arr[:,0].min():.4f}, {q_arr[:,0].max():.4f}]")
print(f"  Q(s, SWITCH) mean={q_arr[:,1].mean():.4f}  std={q_arr[:,1].std():.4f}  range=[{q_arr[:,1].min():.4f}, {q_arr[:,1].max():.4f}]")

margin = q_arr[:,0] - q_arr[:,1]
print(f"\n  Q(KEEP) - Q(SWITCH):  mean={margin.mean():.4f}  std={margin.std():.6f}")
if margin.std() < 0.001:
    print("  *** COLLAPSED: Q-values are nearly identical — agent is indifferent ***")
    print("  *** This means the reward signal is too weak to differentiate actions ***")
elif action_counts[0] > 480:
    print("  *** DEGENERATE: Agent always KEEPs — never learned when to switch ***")
elif action_counts[1] > 480:
    print("  *** DEGENERATE: Agent always SWITCHes — oscillates phases every step ***")
else:
    print("  OK: Agent uses both actions")

print(f"\nQueue statistics over 500 steps:")
print(f"  Mean: {np.mean(queues):.2f}  Std: {np.std(queues):.2f}  Min: {np.min(queues):.0f}  Max: {np.max(queues):.0f}")
if np.std(queues) < 0.5:
    print("  *** WARNING: Queue barely varies — environment may be too deterministic ***")

phase_counts = Counter(phases)
print(f"\nPhase distribution:")
for ph, cnt in sorted(phase_counts.items()):
    print(f"  Phase {ph}: {cnt} steps ({cnt/5:.1f}%)")

print("\n" + "="*55)
print("  Copy this output and share it — tells us exactly what to fix")
print("="*55)
