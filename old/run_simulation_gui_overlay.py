import os, sys, time, torch
from agent import DQNAgent
from environment import TrafficEnv

# Add TraCI path
sys.path.append(r"C:\Program Files\Eclipse\Sumo\tools")
import traci

MODEL_PATH = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files\dqn_model.pth"

SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG = "traffic.sumocfg"

# Load environment
env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)

# DQN settings
state_dim = 5
action_dim = 2
agent = DQNAgent(state_dim, action_dim)
agent.policy_net.load_state_dict(torch.load(MODEL_PATH))
agent.policy_net.eval()

# Start SUMO GUI
state = env.reset(use_gui=True)

# Detect SUMO GUI view ID
view_id = traci.gui.getIDList()[0]     # Example: "View #0"

done = False
step = 0
MAX_STEPS = 1500

while not done and step < MAX_STEPS:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    state = next_state

    # --- OVERLAY UPDATE ---
    traci.gui.setSchema(view_id, "real world")
    text = f"Step: {step} | Phase: {state[-1]:.0f} | Queues: {state[:-1]}"
    traci.gui.setZoom(view_id, 450)
    traci.gui.addLabel(view_id, "label1", text, 10, 40, (255,0,0,255), "xkcd")
    
    time.sleep(0.3)   # slow down simulation
    step += 1

env.close()
print("GUI Simulation Finished.")
