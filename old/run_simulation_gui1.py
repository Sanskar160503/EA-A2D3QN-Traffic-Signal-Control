import torch, numpy as np, time
from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv
 
MODEL_PATH      = "dqn_final.pth"
SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MAX_STEPS       = 1500
STEP_DELAY      = 0.05
 
env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
agent = D3QNAgent(state_dim=11, action_dim=2)
agent.policy_net.load_state_dict(
    torch.load(MODEL_PATH, map_location=agent.device, weights_only=True))
agent.policy_net.eval()
agent.epsilon = 0.0
 
print("\n" + "="*65)
print("  D3QN Traffic Signal Simulation  |  Model:", MODEL_PATH)
print("  Device:", agent.device)
print("="*65)
print("\nSUMO GUI opening — press Play (▶) in SUMO to start.\n")
 
state = env.reset(use_gui=True)
 
print(f"{'Step':>5}  {'Ph':>2}  {'N':>3}{'S':>3}{'E':>3}{'W':>3}  {'TQ':>4}  {'Action':>7}  Note")
print("-"*62)
 
step, done = 0, False
total_q = []
 
while not done and step < MAX_STEPS:
    action = agent.act(state)
 
    q_N   = round(state[0] * 10)
    q_S   = round(state[1] * 10)
    q_E   = round(state[2] * 10)
    q_W   = round(state[3] * 10)
    phase = int(round(state[8]))
    em    = bool(state[10])
    act_s = "KEEP  " if action == 0 else "SWITCH"
    note  = "<<< EMERGENCY + PATH CLEARED" if em else ""
    q_sum = q_N + q_S + q_E + q_W
    total_q.append(q_sum)
 
    if step % 10 == 0:
        print(f"{step:5d}  {phase:2d}  {q_N:3d}{q_S:3d}{q_E:3d}{q_W:3d}  "
              f"{q_sum:4d}  {act_s}  {note}")
 
    next_state, reward, done = env.step(action)
    state = next_state
    step += 1
    time.sleep(STEP_DELAY)
 
env.close()
print("\n" + "="*65)
print(f"  Done — {step} steps")
print(f"  Avg queue: {np.mean(total_q):.2f}  "
      f"Min: {np.min(total_q)}  Max: {np.max(total_q)}")
print("="*65)