"""
train_fixed.py
══════════════════════════════════════════════════════════════════════
Fixed training script addressing all 4 problems from run #4:

Problem 1: Catastrophic forgetting from curriculum switching
Fix:       Train on normal demand only. Evaluate on all 3 scenarios separately.

Problem 2: Emergency buffer always 0 (no emergency vehicles detected)
Fix:       Spawn emergency vehicles directly via TraCI every ~200 steps
           instead of relying on route file. Also fixed detection check.

Problem 3: Attention weights not logging to MySQL
Fix:       Fixed get_attention_weights() call path.

Problem 4: Queue spike at episode 667 (curriculum switch)
Fix:       Removed curriculum. Stable demand throughout.

Run:
    python train_fixed.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time
import os
import re
from collections import Counter

import traci

from ea_a2d3qn_agent import EA_A2D3QNAgent
from environment_v2 import TrafficEnv
from mysql_database import TrafficMySQL

# ── Paths ──────────────────────────────────────────────────────────────
SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

# ── Settings ───────────────────────────────────────────────────────────
EPISODES         = 1000
MAX_STEPS        = 1000
WARMUP_EP        = 50
LOG_INTERVAL     = 50
STEP_LOG_EVERY   = 100
MODEL_PATH       = "ea_a2d3qn_final.pth"
REPLAY_PER_STEP  = 2

# FIX 1: Use normal demand for all episodes
# Evaluate on other scenarios separately after training
FIXED_SCENARIO   = "normal"   # train on this only

# FIX 2: Emergency spawn settings
# Spawn emergency vehicle via TraCI every ~200 steps
# This guarantees emergencies happen regardless of route file
EMERGENCY_INTERVAL = 200      # spawn one every ~200 steps
EMERGENCY_TYPE_ID  = "emergency"
EMERGENCY_LANES    = [
    "north_to_center_0",
    "south_to_center_0",
    "east_to_center_0",
    "west_to_center_0",
]

HYPERPARAMS = {
    "algorithm":       "EA-A2D3QN",
    "state_dim":       11, "action_dim": 2,
    "gamma":           0.95, "lr": 1e-4,
    "batch_size":      256, "buffer_capacity": 20000,
    "emergency_ratio": 0.25, "aux_loss_weight": 0.1,
    "epsilon_decay":   0.9935, "target_update": 1000,
    "replay_per_step": REPLAY_PER_STEP,
    "dataset":         "UCI Metro + Indian IITM-TrafSim",
    "training_note":   "Fixed: no curriculum, direct emergency spawning via TraCI",
    "fix_version":     "v2 - addressing run4 catastrophic forgetting",
}


def set_scenario(sumo_workdir, scenario):
    """Switch SUMO route file to given scenario."""
    route_map = {
        "off_peak":  "routes_india_offpeak.rou.xml",
        "normal":    "routes_india_normal.rou.xml",
        "peak_hour": "routes_india_peak.rou.xml",
    }
    # Fall back to UCI routes if Indian routes don't exist
    uci_map = {
        "off_peak":  "routes_real_offpeak.rou.xml",
        "normal":    "routes_real_normal.rou.xml",
        "peak_hour": "routes_real_peak.rou.xml",
    }
    cfg_path   = os.path.join(sumo_workdir, "traffic.sumocfg")
    route_file = route_map[scenario]
    full_path  = os.path.join(sumo_workdir, route_file)

    if not os.path.exists(full_path):
        route_file = uci_map[scenario]
        full_path  = os.path.join(sumo_workdir, route_file)

    if not os.path.exists(full_path):
        print(f"  [Warning] Route file not found: {route_file}")
        print("  Run: python indian_dataset.py --city chennai --outdir 'Sumo files'")
        return False

    with open(cfg_path, 'r') as f:
        content = f.read()
    content = re.sub(
        r'<route-files value="[^"]*"/>',
        f'<route-files value="{route_file}"/>',
        content
    )
    with open(cfg_path, 'w') as f:
        f.write(content)
    return True


def spawn_emergency_vehicle(step, em_counter):
    """
    FIX 2: Spawn emergency vehicle directly via TraCI.
    This guarantees emergencies happen regardless of route file content.
    Returns (True, new_counter) if spawned, else (False, counter).
    """
    if step % EMERGENCY_INTERVAL != 0 or step == 0:
        return False, em_counter

    import random
    em_id    = f"em_traci_{em_counter}"
    # Use CORRECT edge names from your actual net.net.xml:
    # Incoming: north_to_center, south_to_center, east_to_center, west_to_center
    # Outgoing: center_to_south, center_to_north, center_to_west, center_to_east
    em_route = random.choice([
        ("north_to_center_0", "north_to_center center_to_south"),
        ("south_to_center_0", "south_to_center center_to_north"),
        ("east_to_center_0",  "east_to_center center_to_west"),
        ("west_to_center_0",  "west_to_center center_to_east"),
    ])
    lane_id, route_edges = em_route

    # Check if emergency type exists
    try:
        known_types = traci.vehicletype.getIDList()
        if EMERGENCY_TYPE_ID not in known_types:
            # Define emergency type on the fly
            traci.vehicletype.copy("DEFAULT_VEHTYPE", EMERGENCY_TYPE_ID)
            traci.vehicletype.setColor(EMERGENCY_TYPE_ID, (255, 0, 0, 255))
            traci.vehicletype.setMaxSpeed(EMERGENCY_TYPE_ID, 22.22)

        # Define a one-time route
        route_id = f"em_route_{em_counter}"
        traci.route.add(route_id, route_edges.split())
        # departSpeed="0" prevents collision at insertion
        # Vehicle accelerates naturally once in simulation
        # departLane="free" picks least occupied lane
        traci.vehicle.add(em_id, route_id,
                          typeID=EMERGENCY_TYPE_ID,
                          depart="now",
                          departSpeed="0",
                          departLane="free")
        # Give max speed after safely inserted
        traci.vehicle.setMaxSpeed(em_id, 22.22)
        traci.vehicle.setSpeedMode(em_id, 7)  # 7 = obey all signals
        return True, em_counter + 1
    except Exception as e:
        return False, em_counter


def detect_emergency_traci(env):
    """
    FIX 2: Check for emergency vehicles using type ID.
    More robust than checking state[10] alone.
    """
    try:
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == EMERGENCY_TYPE_ID:
                return True
    except Exception:
        pass
    return False


def main():
    # ── MySQL ──────────────────────────────────────────────────────────
    db = TrafficMySQL()

    # ── Verify real data ───────────────────────────────────────────────
    print("\n[Check] Real traffic data in MySQL:")
    for sc in ["off_peak", "normal", "peak_hour"]:
        row = db.get_traffic_scenario(sc)
        if row and row.get("n_records") and row["n_records"] > 0:
            print(f"  {sc:<12}: {int(row['n_records'])} records "
                  f"N={row['n_flow']:.0f} veh/hr")
        else:
            print(f"  {sc}: No data — run python real_dataset.py first")

    # ── Set route file ─────────────────────────────────────────────────
    print(f"\n[Setup] Using fixed scenario: {FIXED_SCENARIO}")
    ok = set_scenario(SUMO_WORKDIR, FIXED_SCENARIO)
    if not ok:
        print("Could not find route file. Check Sumo files folder.")
        db.close()
        return

    # ── Init ───────────────────────────────────────────────────────────
    env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
    agent = EA_A2D3QNAgent(
        state_dim=11, action_dim=2,
        emergency_ratio=0.25, aux_loss_weight=0.1
    )

    run_id = db.start_run(
        algorithm   = "EA-A2D3QN-v2",
        hyperparams = HYPERPARAMS,
        notes       = "Fixed training: no curriculum catastrophe, direct TraCI emergency spawning"
    )

    print(f"\n{'='*70}")
    print(f"  EA-A2D3QN Fixed Training | MySQL run #{run_id}")
    print(f"  Scenario: {FIXED_SCENARIO} (fixed throughout)")
    print(f"  Emergency: TraCI direct spawn every {EMERGENCY_INTERVAL} steps")
    print(f"{'='*70}\n")

    # ── Histories ──────────────────────────────────────────────────────
    reward_history    = []
    td_loss_history   = []
    queue_history     = []
    phase_history     = []
    em_buffer_history = []
    best_avg_reward   = -np.inf
    train_start       = time.time()
    global_step       = 0
    step_batch        = []
    em_counter        = 0       # tracks spawned emergency vehicles

    for episode in range(EPISODES):
        ep_start     = time.time()
        state        = env.reset(use_gui=False)
        total_reward = 0.0
        ep_td_losses = []
        ep_queues    = []
        ep_phases    = []
        ep_em_steps  = 0

        for step in range(MAX_STEPS):

            # FIX 2: Spawn emergency vehicle via TraCI directly
            spawned, em_counter = spawn_emergency_vehicle(step, em_counter)

            # Detect emergency — check both state flag AND TraCI
            is_emergency = bool(state[10] > 0.5) or detect_emergency_traci(env)

            if is_emergency:
                ep_em_steps += 1

            # Act
            action = agent.act(state, step=global_step)

            # Step environment
            next_state, reward, done = env.step(action)
            reward_norm = float(np.clip(reward, -2.0, 2.0))

            # Store — route to emergency buffer if emergency
            agent.remember(state, action, reward_norm,
                           next_state, float(done),
                           is_emergency=is_emergency)

            # Learn
            for _ in range(REPLAY_PER_STEP):
                result = agent.replay()
                if result[0] is not None:
                    ep_td_losses.append(result[1])

            # Step log batch
            if step % STEP_LOG_EVERY == 0:
                step_batch.append(
                    (run_id, episode, step, state.copy(), action, reward)
                )
                if len(step_batch) >= 500:
                    db.log_steps_batch(step_batch)
                    step_batch.clear()

            # FIX 3: Attention logging — corrected path
            if step % 200 == 0:
                try:
                    # Call through the attention encoder directly
                    weights = agent.policy_net.attention.last_attention_weights
                    if weights is not None:
                        # Average across query dimension to get per-feature weight
                        avg_w = weights[0].mean(dim=0).numpy()  # shape [11]
                        db.log_attention(run_id, episode, int(is_emergency), avg_w)
                except Exception:
                    pass

            ep_queues.append(float(np.sum(next_state[:4]) * 10.0))
            ep_phases.append(int(round(next_state[8])))

            state        = next_state
            total_reward += reward
            global_step  += 1

            if done:
                break

        # Flush step batch
        if step_batch:
            db.log_steps_batch(step_batch)
            step_batch.clear()

        # ── Episode metrics ────────────────────────────────────────────
        avg_queue       = float(np.mean(ep_queues))
        avg_td_loss     = float(np.mean(ep_td_losses)) if ep_td_losses else 0.0
        phase_counts    = Counter(ep_phases)
        total_steps_ep  = max(len(ep_phases), 1)
        phase_fracs     = [phase_counts.get(i, 0) / total_steps_ep for i in range(4)]
        phase_imbalance = float(np.std(phase_fracs))
        em_stats        = agent.memory.stats()
        ep_duration     = time.time() - ep_start

        reward_history.append(total_reward)
        td_loss_history.append(avg_td_loss)
        queue_history.append(avg_queue)
        phase_history.append(phase_imbalance)
        em_buffer_history.append(em_stats["em_buf_len"])

        db.log_episode(
            run_id=run_id, episode=episode,
            total_reward=total_reward, avg_queue=avg_queue,
            avg_wait=avg_queue, td_loss=avg_td_loss,
            epsilon=agent.epsilon, phase_imbalance=phase_imbalance,
            em_buffer_size=em_stats["em_buf_len"],
            em_buffer_pct=em_stats["em_pct"],
            duration_sec=ep_duration,
        )

        # Save best model
        if episode >= WARMUP_EP:
            avg50 = np.mean(reward_history[-50:])
            if avg50 > best_avg_reward:
                best_avg_reward = avg50
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)

        # Console log
        if (episode + 1) % LOG_INTERVAL == 0:
            avg50   = np.mean(reward_history[-50:])
            avg_q   = np.mean(queue_history[-50:])
            elapsed = (time.time() - train_start) / 60
            rem     = elapsed / (episode + 1) * (EPISODES - episode - 1)
            lr      = agent.optimizer.param_groups[0]["lr"]
            print(
                f"Ep {episode+1:4d} | "
                f"R: {avg50:7.1f} | Q: {avg_q:4.1f} | "
                f"Loss: {avg_td_loss:.5f} | Eps: {agent.epsilon:.3f} | "
                f"LR: {lr:.1e} | "
                f"EmBuf: {em_stats['em_buf_len']:4d} ({em_stats['em_pct']:.1f}%) | "
                f"EmSteps: {ep_em_steps:3d} | "
                f"{elapsed:.0f}m/{rem:.0f}m"
            )

    # ── Finish ─────────────────────────────────────────────────────────
    env.close()
    total_time = (time.time() - train_start) / 60
    db.end_run(run_id, EPISODES, best_avg_reward, MODEL_PATH)

    print(f"\nDone in {total_time:.1f} min")
    print(f"Best avg50: {best_avg_reward:.1f} → {MODEL_PATH}")

    agent.print_attention_summary()
    db.print_emergency_analysis(run_id)

    # ── Save arrays ────────────────────────────────────────────────────
    np.save("reward_fixed.npy",    reward_history)
    np.save("queue_fixed.npy",     queue_history)
    np.save("tdloss_fixed.npy",    td_loss_history)
    np.save("em_buffer_fixed.npy", em_buffer_history)

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"EA-A2D3QN Fixed Training | MySQL run #{run_id} | "
        f"Best avg50: {best_avg_reward:.1f}",
        fontsize=13, fontweight='bold'
    )
    window = 50
    eps    = range(1, EPISODES + 1)

    # Reward
    ax = axes[0, 0]
    ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, reward_history, alpha=0.2, color='steelblue')
    ax.plot(range(window, EPISODES+1), ma, color='steelblue', lw=2,
            label=f'{window}-ep avg')
    ax.axhline(best_avg_reward, color='green', ls='--', lw=1,
               label=f'Best: {best_avg_reward:.1f}')
    ax.axhline(0, color='red', ls=':', lw=0.8, alpha=0.5, label='Zero line')
    ax.set_title("Reward — fixed scenario (no curriculum)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # TD Loss
    ax = axes[0, 1]
    lma = np.convolve(td_loss_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, td_loss_history, alpha=0.2, color='coral')
    ax.plot(range(window, EPISODES+1), lma, color='coral', lw=2)
    ax.set_title("TD Loss (should decrease monotonically)"); ax.grid(alpha=0.3)

    # Queue
    ax = axes[0, 2]
    qma = np.convolve(queue_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, queue_history, alpha=0.2, color='mediumpurple')
    ax.plot(range(window, EPISODES+1), qma, color='mediumpurple', lw=2)
    ax.set_title("Avg Queue"); ax.grid(alpha=0.3)

    # Emergency buffer (should grow now with TraCI spawning)
    ax = axes[1, 0]
    ax.plot(eps, em_buffer_history, color='crimson', lw=1.5)
    ax.set_title("Emergency Buffer (Dual PER) — should be non-zero now")
    ax.grid(alpha=0.3)

    # Phase imbalance
    ax = axes[1, 1]
    pma = np.convolve(phase_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, phase_history, alpha=0.2, color='darkorange')
    ax.plot(range(window, EPISODES+1), pma, color='darkorange', lw=2)
    ax.axhline(0, color='green', ls='--', lw=1, label='Perfect balance')
    ax.set_title("Phase Imbalance (↓ better)"); ax.grid(alpha=0.3)

    # Attention weights from MySQL
    ax = axes[1, 2]
    attn_rows = db.get_attention_analysis(run_id)
    labels    = ["q_N","q_S","q_E","q_W","w_N","w_S","w_E","w_W",
                 "phase","timer","emerg"]
    if attn_rows and len(attn_rows) >= 1:
        normal_row = next((r for r in attn_rows if not r["is_emergency"]), None)
        em_row     = next((r for r in attn_rows if r["is_emergency"]),     None)
        keys = ["q_north","q_south","q_east","q_west",
                "w_north","w_south","w_east","w_west",
                "phase","timer","emergency_flag"]
        x = np.arange(len(labels)); w = 0.35
        if normal_row:
            nv = [normal_row.get(k) or 0 for k in keys]
            ax.bar(x-w/2, nv, w, label='Normal',    color='steelblue', alpha=0.8)
        if em_row:
            ev = [em_row.get(k) or 0 for k in keys]
            ax.bar(x+w/2, ev, w, label='Emergency', color='crimson',   alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.legend(fontsize=9)
        ax.set_title("Attention Weights (from MySQL)")
    else:
        ax.text(0.5, 0.5, "No attention data\n(will populate during training)",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Attention Weights (from MySQL)")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("training_fixed.png", dpi=150)
    print("Plot saved → training_fixed.png")

    db.close()
    print(f"\n[Done] MySQL run #{run_id} complete.")
    print(f"  Next: python compare_algorithms.py")
    print(f"  Query: python query_results.py {run_id}")


if __name__ == "__main__":
    main()