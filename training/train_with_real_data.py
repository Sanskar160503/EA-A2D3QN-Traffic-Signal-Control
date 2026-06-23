"""
train_with_real_data.py
══════════════════════════════════════════════════════════════════════
Training script that pulls real traffic demand from MySQL
and trains EA-A2D3QN across 3 real-world scenarios.

Run order:
  1. python real_dataset.py          ← downloads UCI dataset, loads to MySQL
  2. python train_with_real_data.py  ← trains using real demand from MySQL
  3. python query_results.py 1       ← inspect results
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time
from collections import Counter

from ea_a2d3qn_agent import EA_A2D3QNAgent
from environment_v2 import TrafficEnv
from mysql_database import TrafficMySQL

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"

EPISODES        = 1000
MAX_STEPS       = 1000
WARMUP_EP       = 100
LOG_INTERVAL    = 50
STEP_LOG_EVERY  = 100
MODEL_PATH      = "ea_a2d3qn_final.pth"
REPLAY_PER_STEP = 2

HYPERPARAMS = {
    "algorithm":       "EA-A2D3QN",
    "state_dim":       11, "action_dim": 2,
    "gamma":           0.95, "lr": 1e-4,
    "batch_size":      256, "buffer_capacity": 20000,
    "emergency_ratio": 0.25, "aux_loss_weight": 0.1,
    "epsilon_decay":   0.9935, "target_update": 1000,
    "dataset":         "UCI Metro Interstate Traffic Volume (I-94, 2012-2018)",
    "dataset_source":  "UCI ML Repository — real traffic data",
}

# ── Scenario rotation: train on mix of real demand levels ──────────────
# Episodes 0-333: normal demand (most common real scenario)
# Episodes 334-666: mix of all three
# Episodes 667-999: weighted toward peak (hardest case)
def get_scenario_for_episode(episode):
    if episode < 334:
        return "normal"
    elif episode < 667:
        return ["normal", "off_peak", "peak_hour"][episode % 3]
    else:
        return ["peak_hour", "normal", "peak_hour"][episode % 3]


def switch_sumo_route(sumo_workdir, scenario):
    """Update traffic.sumocfg to use the real dataset route file."""
    route_map = {
        "off_peak":  "routes_real_offpeak.rou.xml",
        "normal":    "routes_real_normal.rou.xml",
        "peak_hour": "routes_real_peak.rou.xml",
    }
    cfg_path   = os.path.join(sumo_workdir, "traffic.sumocfg")
    route_file = route_map[scenario]

    with open(cfg_path, 'r') as f:
        content = f.read()
    content = re.sub(
        r'<route-files value="[^"]*"/>',
        f'<route-files value="{route_file}"/>',
        content
    )
    with open(cfg_path, 'w') as f:
        f.write(content)


def main():
    # ── Connect to MySQL ───────────────────────────────────────────────
    db = TrafficMySQL()

    # ── Verify real data is in MySQL ───────────────────────────────────
    print("\n[Check] Real traffic data in MySQL:")
    for sc in ["off_peak", "normal", "peak_hour"]:
        row = db.get_traffic_scenario(sc)
        if row and row["n_records"] and row["n_records"] > 0:
            print(f"  {sc:<12}: N={row['n_flow']:.0f} S={row['s_flow']:.0f} "
                  f"E={row['e_flow']:.0f} W={row['w_flow']:.0f} veh/hr "
                  f"({int(row['n_records'])} records from UCI dataset)")
        else:
            print(f"  {sc}: NO DATA FOUND")
            print("  → Run: python real_dataset.py")
            db.close()
            return

    # ── Verify real route files exist ──────────────────────────────────
    route_files = [
        "routes_real_offpeak.rou.xml",
        "routes_real_normal.rou.xml",
        "routes_real_peak.rou.xml",
    ]
    for rf in route_files:
        full = os.path.join(SUMO_WORKDIR, rf)
        if not os.path.exists(full):
            print(f"\n  Route file missing: {full}")
            print("  → Run: python real_dataset.py --outdir \"Sumo files\"")
            db.close()
            return
    print("\n[Check] All real route files found. Starting training.\n")

    # ── Init agent ─────────────────────────────────────────────────────
    env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
    agent = EA_A2D3QNAgent(
        state_dim=11, action_dim=2,
        emergency_ratio=0.25, aux_loss_weight=0.1
    )

    run_id = db.start_run(
        algorithm   = "EA-A2D3QN",
        hyperparams = HYPERPARAMS,
        notes       = (
            "Training on UCI Metro Interstate Traffic Volume dataset "
            "(I-94 Minnesota 2012-2018). Real demand: off_peak=1082, "
            "normal=2646, peak_hour=5239 veh/hr. "
            "3-scenario curriculum: normal → mixed → peak."
        )
    )

    # ── Training loop ──────────────────────────────────────────────────
    reward_history    = []
    td_loss_history   = []
    queue_history     = []
    phase_history     = []
    em_buffer_history = []
    scenario_history  = []
    best_avg_reward   = -np.inf
    train_start       = time.time()
    global_step       = 0
    step_batch        = []
    current_scenario  = None

    print("="*70)
    print(f"  EA-A2D3QN | Real Dataset Training | MySQL run #{run_id}")
    print("="*70 + "\n")

    for episode in range(EPISODES):
        ep_start  = time.time()

        # Switch SUMO demand to real scenario for this episode
        scenario  = get_scenario_for_episode(episode)
        if scenario != current_scenario:
            switch_sumo_route(SUMO_WORKDIR, scenario)
            current_scenario = scenario

        scenario_history.append(scenario)
        state        = env.reset(use_gui=False)
        total_reward = 0.0
        ep_td_losses = []
        ep_queues    = []
        ep_phases    = []

        for step in range(MAX_STEPS):
            action       = agent.act(state, step=global_step)
            next_state, reward, done = env.step(action)
            is_emergency = bool(state[10] > 0.5)
            reward_norm  = float(np.clip(reward, -2.0, 2.0))

            agent.remember(state, action, reward_norm,
                           next_state, float(done),
                           is_emergency=is_emergency)

            for _ in range(REPLAY_PER_STEP):
                result = agent.replay()
                if result[0] is not None:
                    ep_td_losses.append(result[1])

            if step % STEP_LOG_EVERY == 0:
                step_batch.append(
                    (run_id, episode, step, state.copy(), action, reward)
                )
                if len(step_batch) >= 500:
                    db.log_steps_batch(step_batch)
                    step_batch.clear()

            if step % 300 == 0:
                weights = agent.policy_net.get_attention_weights()
                if weights is not None:
                    avg_w = weights[0].mean(axis=0).numpy()
                    db.log_attention(run_id, episode, is_emergency, avg_w)

            ep_queues.append(float(np.sum(next_state[:4]) * 10.0))
            ep_phases.append(int(round(next_state[8])))

            state        = next_state
            total_reward += reward
            global_step  += 1
            if done:
                break

        if step_batch:
            db.log_steps_batch(step_batch)
            step_batch.clear()

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

        if episode >= WARMUP_EP:
            avg50 = np.mean(reward_history[-50:])
            if avg50 > best_avg_reward:
                best_avg_reward = avg50
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)

        if (episode + 1) % LOG_INTERVAL == 0:
            avg50   = np.mean(reward_history[-50:])
            avg_q   = np.mean(queue_history[-50:])
            elapsed = (time.time() - train_start) / 60
            rem     = elapsed / (episode + 1) * (EPISODES - episode - 1)
            print(
                f"Ep {episode+1:4d} [{scenario:<9}] | "
                f"R: {avg50:7.1f} | Q: {avg_q:4.1f} | "
                f"Loss: {avg_td_loss:.5f} | "
                f"EmBuf: {em_stats['em_buf_len']:4d} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"{elapsed:.0f}m/{rem:.0f}m"
            )

    # ── Finish ──────────────────────────────────────────────────────────
    env.close()
    total_time = (time.time() - train_start) / 60
    db.end_run(run_id, EPISODES, best_avg_reward, MODEL_PATH)

    print(f"\nDone in {total_time:.1f} min | Best → {MODEL_PATH}")

    # ── Emergency and attention analysis from MySQL ────────────────────
    agent.print_attention_summary()
    db.print_emergency_analysis(run_id)

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"EA-A2D3QN | UCI Real Dataset | MySQL run #{run_id} | "
        f"Best: {best_avg_reward:.1f}",
        fontsize=13, fontweight='bold'
    )
    window = 50
    eps    = range(1, EPISODES + 1)

    sc_colors = {"off_peak": "skyblue", "normal": "steelblue", "peak_hour": "darkblue"}

    ax = axes[0, 0]
    for sc in ["off_peak", "normal", "peak_hour"]:
        idx = [i for i, s in enumerate(scenario_history) if s == sc]
        ax.scatter(idx, [reward_history[i] for i in idx],
                   alpha=0.1, s=2, color=sc_colors[sc], label=sc)
    ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    ax.plot(range(window, EPISODES+1), ma, color='white' if False else 'navy',
            lw=2, label=f'{window}-ep avg')
    ax.axhline(best_avg_reward, color='green', ls='--', lw=1)
    ax.set_title("Reward (coloured by real traffic scenario)")
    ax.legend(fontsize=8, markerscale=5); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    lma = np.convolve(td_loss_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, td_loss_history, alpha=0.2, color='coral')
    ax.plot(range(window, EPISODES+1), lma, color='coral', lw=2)
    ax.set_title("TD Loss"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    qma = np.convolve(queue_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, queue_history, alpha=0.2, color='mediumpurple')
    ax.plot(range(window, EPISODES+1), qma, color='mediumpurple', lw=2)
    ax.set_title("Avg Queue"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(eps, em_buffer_history, color='crimson', lw=1.5)
    ax.set_title("Emergency Buffer (Dual PER — Novel)"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    pma = np.convolve(phase_history, np.ones(window)/window, mode='valid')
    ax.plot(eps, phase_history, alpha=0.2, color='darkorange')
    ax.plot(range(window, EPISODES+1), pma, color='darkorange', lw=2)
    ax.axhline(0, color='green', ls='--', lw=1)
    ax.set_title("Phase Imbalance"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    attn_rows = db.get_attention_analysis(run_id)
    labels    = ["q_N","q_S","q_E","q_W","w_N","w_S","w_E","w_W",
                 "phase","timer","emerg"]
    if attn_rows and len(attn_rows) >= 2:
        normal_row = next((r for r in attn_rows if not r["is_emergency"]), None)
        em_row     = next((r for r in attn_rows if r["is_emergency"]),     None)
        if normal_row and em_row:
            keys = ["q_north","q_south","q_east","q_west",
                    "w_north","w_south","w_east","w_west",
                    "phase","timer","emergency_flag"]
            nv = [normal_row[k] or 0 for k in keys]
            ev = [em_row[k]     or 0 for k in keys]
            x  = np.arange(len(labels))
            w  = 0.35
            ax.bar(x-w/2, nv, w, label='Normal',    color='steelblue', alpha=0.8)
            ax.bar(x+w/2, ev, w, label='Emergency', color='crimson',   alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.legend(fontsize=9)
    ax.set_title("Attention Weights from MySQL"); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("training_real_data_final.png", dpi=150)
    print("Plot saved → training_real_data_final.png")

    db.close()
    print(f"\n[Done] MySQL run #{run_id} complete.")
    print(f"  Query: python query_results.py {run_id}")


if __name__ == "__main__":
    main()
