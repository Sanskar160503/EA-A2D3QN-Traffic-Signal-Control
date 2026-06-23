#run_simulation.py
from agent import DQNAgent
from environment import TrafficEnv
import torch
import time

MODEL_PATH = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files\dqn_model.pth"
SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG = "traffic.sumocfg"

env = TrafficEnv(
    SUMO_BINARY,
    SUMO_GUI_BINARY,
    SUMO_CFG,
    SUMO_WORKDIR
)

agent = DQNAgent(state_dim=6, action_dim=2)
agent.policy_net.load_state_dict(torch.load(MODEL_PATH))
agent.policy_net.eval()

state = env.reset(use_gui=True)

print("\n=== RL Traffic Control Simulation Started ===\n")

step = 0
done = False

while not done and step < 1500:
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    queues = state[:-1]
    phase = state[-1]
    action_name = "KEEP" if action == 0 else "SWITCH"

    print(
        f"[STEP {step:04d}] "
        f"Phase:{int(phase)} | "
        f"Queues N:{int(queues[0])} S:{int(queues[1])} "
        f"E:{int(queues[2])} W:{int(queues[3])} | "
        f"Action:{action_name} | "
        f"Reward:{reward:.2f}"
    )

    state = next_state
    step += 1
    time.sleep(0.2)

    if env.detect_emergency():
       print("🚑 Emergency vehicle detected – priority granted")


env.close()
print("\n=== Simulation Finished Cleanly ===")
