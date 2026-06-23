"""
run_simulation_gui.py
══════════════════════════════════════════════════════════════════════
Live SUMO GUI simulation using the trained EA-A2D3QN model.

Controls:
  - SUMO GUI opens automatically
  - Press Play (▶) in SUMO to start vehicles moving
  - Console prints step-by-step actions, queues, emergency events
  - Press Ctrl+C to stop early

The script tries ea_a2d3qn_final.pth first (novel model),
falls back to dqn_final.pth (original D3QN) if not found.
"""

import torch
import numpy as np
import time
import os

import traci

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MAX_STEPS       = 1500
STEP_DELAY      = 0.05    # seconds between steps (reduce to speed up, 0 = fastest)
PRINT_EVERY     = 10      # print console output every N steps

# ── Load best available model ──────────────────────────────────────────
EA_MODEL  = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files\ea_a2d3qn_final.pth"
D3QN_MODEL = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files\dqn_final.pth"

# Try EA-A2D3QN first, fall back to D3QN
if os.path.exists(EA_MODEL):
    MODEL_PATH  = EA_MODEL
    MODEL_NAME  = "EA-A2D3QN (Novel)"
    USE_EA      = True
elif os.path.exists(D3QN_MODEL):
    MODEL_PATH  = D3QN_MODEL
    MODEL_NAME  = "D3QN (Baseline)"
    USE_EA      = False
else:
    # Also check current folder
    for path, name, ea in [
        ("ea_a2d3qn_final.pth", "EA-A2D3QN (Novel)", True),
        ("dqn_final.pth", "D3QN (Baseline)", False),
    ]:
        if os.path.exists(path):
            MODEL_PATH = path
            MODEL_NAME = name
            USE_EA     = ea
            break
    else:
        print("ERROR: No model file found.")
        print("Expected: ea_a2d3qn_final.pth or dqn_final.pth")
        print("Run training first: python train_fixed.py")
        exit(1)

# ── Load agent ─────────────────────────────────────────────────────────
if USE_EA:
    from ea_a2d3qn_agent import EA_A2D3QNAgent
    agent = EA_A2D3QNAgent(state_dim=11, action_dim=2)
    agent.policy_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device, weights_only=True)
    )
else:
    from agent_v3 import D3QNAgent
    agent = D3QNAgent(state_dim=11, action_dim=2)
    agent.policy_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device, weights_only=True)
    )

agent.policy_net.eval()
agent.epsilon = 0.0   # fully greedy — no random exploration

from environment_v2 import TrafficEnv
env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)

# ── Header ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  EA-A2D3QN Live Simulation")
print(f"  Model:  {MODEL_NAME}")
print(f"  File:   {MODEL_PATH}")
print(f"  Device: {agent.device}")
print("="*70)
print("\n  SUMO GUI is opening...")
print("  → Press Play (▶) in the SUMO window to start vehicles")
print("  → Press Ctrl+C here to stop early\n")

# ── Start simulation ───────────────────────────────────────────────────
state = env.reset(use_gui=True)

print(f"{'Step':>5}  {'Phase':>5}  {'Timer':>5}  "
      f"{'N':>4}{'S':>4}{'E':>4}{'W':>4}  "
      f"{'Total':>5}  {'Action':>7}  Status")
print("─" * 72)

step          = 0
done          = False
total_q       = []
em_events     = 0
em_start      = None
em_clear_times = []
switch_count  = 0
keep_count    = 0
ep_start      = time.time()
em_counter    = 0
EM_INTERVAL   = 300   # spawn one emergency every 300 steps = 5 per run

EM_ROUTES = [
    ("north_to_center center_to_south", "emergency_ns"),
    ("south_to_center center_to_north", "emergency_sn"),
    ("east_to_center center_to_west",   "emergency_ew"),
    ("west_to_center center_to_east",   "emergency_we"),
]

def spawn_em(step_num, counter):
    """Guaranteed emergency spawn every EM_INTERVAL steps."""
    if step_num % EM_INTERVAL != 0 or step_num == 0:
        return counter
    import random
    edges, route_id = random.choice(EM_ROUTES)
    vid = f"em_gui_{counter}"
    try:
        traci.vehicle.add(vehID=vid, routeID=route_id,
                          typeID="emergency", depart="now",
                          departSpeed="0", departLane="free")
    except Exception:
        try:
            rid = f"em_r_{counter}"
            traci.route.add(rid, edges.split())
            traci.vehicle.add(vehID=vid, routeID=rid,
                              typeID="emergency", depart="now",
                              departSpeed="0", departLane="free")
        except Exception as e:
            print(f"  [Em spawn failed: {e}]")
            return counter
    try:
        traci.vehicle.setMaxSpeed(vid, 22.22)
    except Exception:
        pass
    print(f"\n  *** EMERGENCY SPAWNED: {vid} via {route_id} ***\n")
    return counter + 1

try:
    while not done and step < MAX_STEPS:

        # Guaranteed emergency spawn
        em_counter = spawn_em(step, em_counter)

        # Agent decides action
        action = agent.act(state)

        # Read state values
        q_N        = round(state[0] * 10)
        q_S        = round(state[1] * 10)
        q_E        = round(state[2] * 10)
        q_W        = round(state[3] * 10)
        phase      = int(round(state[8]))
        timer_pct  = int(state[9] * 100)
        em_flag    = bool(state[10] > 0.5)

        # Also check TraCI directly — catches em_gui_ vehicles
        if not em_flag:
            try:
                for veh_id in traci.vehicle.getIDList():
                    if traci.vehicle.getTypeID(veh_id) == "emergency":
                        em_flag = True
                        break
            except Exception:
                pass

        act_str    = "SWITCH" if action == 1 else "KEEP  "
        q_total    = q_N + q_S + q_E + q_W
        total_q.append(q_total)

        # Track actions
        if action == 1:
            switch_count += 1
        else:
            keep_count += 1

        # Track emergency events
        if em_flag and em_start is None:
            em_start = step
            em_events += 1
        elif not em_flag and em_start is not None:
            em_clear_times.append(step - em_start)
            em_start = None

        # Status string
        if em_flag:
            status = "<<< EMERGENCY — phase override active"
        elif action == 1:
            status = "switching phase"
        else:
            status = ""

        # Print every PRINT_EVERY steps or on emergency
        if step % PRINT_EVERY == 0 or em_flag:
            print(f"{step:5d}  "
                  f"Ph:{phase:1d}  "
                  f"T:{timer_pct:3d}%  "
                  f"{q_N:4d}{q_S:4d}{q_E:4d}{q_W:4d}  "
                  f"{q_total:5d}  "
                  f"{act_str}  "
                  f"{status}")

        # Step environment
        next_state, reward, done = env.step(action)
        state = next_state
        step += 1
        time.sleep(STEP_DELAY)

except KeyboardInterrupt:
    print("\n  Stopped by user.")

# ── Final summary ──────────────────────────────────────────────────────
env.close()
elapsed = time.time() - ep_start

print("\n" + "="*70)
print(f"  Simulation Complete")
print(f"  Steps run:       {step}")
print(f"  Wall time:       {elapsed:.1f}s")
print()
print(f"  Queue statistics:")
print(f"    Average:       {np.mean(total_q):.2f} vehicles")
print(f"    Minimum:       {np.min(total_q)} vehicles")
print(f"    Maximum:       {np.max(total_q)} vehicles")
print()
print(f"  Signal decisions:")
print(f"    KEEP:          {keep_count} ({keep_count/max(step,1)*100:.1f}%)")
print(f"    SWITCH:        {switch_count} ({switch_count/max(step,1)*100:.1f}%)")
print()
print(f"  Emergency events:")
print(f"    Detected:      {em_events}")
if em_clear_times:
    print(f"    Avg clearance: {np.mean(em_clear_times):.1f} steps")
    print(f"    All cleared:   {'Yes' if em_start is None else 'One still active'}")
else:
    print(f"    None occurred during this run")
print("="*70)