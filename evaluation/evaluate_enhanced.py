"""
evaluate_enhanced.py — Enhanced evaluation with 3 baselines
=============================================================
Compares D3QN against:
  1. Fixed-time (30s per phase)
  2. SOTL — Self-Organizing Traffic Lights (built-in SUMO adaptive controller)
  3. Actuated control (SUMO built-in, extends green based on detector)

This addresses the reviewer concern about baseline strength.
Run AFTER setting up real demand routes.
"""

import torch
import numpy as np
import traci
import subprocess
import os
from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MODEL_PATH      = "dqn_final.pth"
EVAL_EPISODES   = 10
MAX_STEPS       = 1000
TL_ID           = "n_center"


def run_rl_episode(env, agent):
    state = env.reset(use_gui=False)
    metrics = {"queue": [], "wait": [], "passed": 0}
    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, _, done = env.step(action)
        metrics["queue"].append(float(np.sum(next_state[:4] * 10)))
        metrics["wait"].append(sum(traci.lane.getLastStepHaltingNumber(l)
                                   for l in env.lanes))
        metrics["passed"] += traci.simulation.getArrivedNumber()
        state = next_state
        if done:
            break
    return metrics


def run_baseline_episode(sumo_binary, sumo_cfg, workdir,
                         mode="fixed", fixed_interval=30):
    """
    Run a full episode under a non-RL controller.
    mode options: 'fixed', 'sotl', 'actuated'
    """
    os.chdir(workdir)
    lanes = ["north_to_center_0", "south_to_center_0",
             "east_to_center_0",  "west_to_center_0"]

    cmd = [sumo_binary, "-c", sumo_cfg, "--start", "--quit-on-end",
           "--no-warnings"]

    if mode == "sotl":
        # SOTL: Self-Organizing Traffic Lights — SUMO built-in adaptive
        cmd += ["--tls.default-type", "sotl_phase"]
    elif mode == "actuated":
        # Actuated: extends green based on detector occupancy
        cmd += ["--tls.default-type", "actuated"]

    traci.start(cmd)
    traci.simulationStep()

    metrics = {"queue": [], "wait": [], "passed": 0}
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
        if mode == "fixed":
            if step % fixed_interval == 0:
                current = traci.trafficlight.getPhase(TL_ID)
                logic   = traci.trafficlight.getCompleteRedYellowGreenDefinition(TL_ID)
                traci.trafficlight.setPhase(TL_ID, (current + 1) % len(logic[0].phases))
        # sotl and actuated: SUMO handles it automatically

        traci.simulationStep()
        q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        w = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        metrics["queue"].append(float(q))
        metrics["wait"].append(float(w))
        metrics["passed"] += traci.simulation.getArrivedNumber()
        step += 1

    traci.close()
    return metrics


def summarize(results):
    return {
        "avg_queue":    np.mean([np.mean(r["queue"])  for r in results]),
        "avg_wait":     np.mean([np.mean(r["wait"])   for r in results]),
        "throughput":   np.mean([r["passed"]           for r in results]),
    }


def main():
    # ── D3QN ──────────────────────────────────────────────────────────────
    env   = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
    agent = D3QNAgent(state_dim=11, action_dim=2)
    agent.policy_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device, weights_only=True)
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0

    print("\n" + "="*65)
    print("  Enhanced Evaluation: D3QN vs 3 Baselines")
    print("="*65)

    print(f"\n[1/4] Running D3QN ({EVAL_EPISODES} episodes)...")
    rl_results = [run_rl_episode(env, agent) for _ in range(EVAL_EPISODES)]
    env.close()

    print(f"[2/4] Running Fixed-time ({EVAL_EPISODES} episodes)...")
    fixed_results = []
    for ep in range(EVAL_EPISODES):
        m = run_baseline_episode(SUMO_BINARY, SUMO_CFG, SUMO_WORKDIR, "fixed")
        fixed_results.append(m)
        print(f"  Ep {ep+1}: queue={np.mean(m['queue']):.2f}")

    print(f"[3/4] Running SOTL ({EVAL_EPISODES} episodes)...")
    sotl_results = []
    for ep in range(EVAL_EPISODES):
        m = run_baseline_episode(SUMO_BINARY, SUMO_CFG, SUMO_WORKDIR, "sotl")
        sotl_results.append(m)
        print(f"  Ep {ep+1}: queue={np.mean(m['queue']):.2f}")

    print(f"[4/4] Running Actuated ({EVAL_EPISODES} episodes)...")
    act_results = []
    for ep in range(EVAL_EPISODES):
        m = run_baseline_episode(SUMO_BINARY, SUMO_CFG, SUMO_WORKDIR, "actuated")
        act_results.append(m)
        print(f"  Ep {ep+1}: queue={np.mean(m['queue']):.2f}")

    # ── Summary table ──────────────────────────────────────────────────────
    rl_s   = summarize(rl_results)
    fix_s  = summarize(fixed_results)
    sotl_s = summarize(sotl_results)
    act_s  = summarize(act_results)

    def pct(base, new):
        return f"{(base-new)/base*100:+.1f}%"

    print("\n\n" + "="*72)
    print(f"  {'Metric':<26} {'D3QN':>9} {'Fixed':>9} {'SOTL':>9} {'Actuated':>9}")
    print("  " + "-"*68)
    print(f"  {'Avg queue length (veh)':<26} "
          f"{rl_s['avg_queue']:>9.2f} {fix_s['avg_queue']:>9.2f} "
          f"{sotl_s['avg_queue']:>9.2f} {act_s['avg_queue']:>9.2f}")
    print(f"  {'Avg wait (veh/step)':<26} "
          f"{rl_s['avg_wait']:>9.2f} {fix_s['avg_wait']:>9.2f} "
          f"{sotl_s['avg_wait']:>9.2f} {act_s['avg_wait']:>9.2f}")
    print(f"  {'Throughput (veh/episode)':<26} "
          f"{rl_s['throughput']:>9.1f} {fix_s['throughput']:>9.1f} "
          f"{sotl_s['throughput']:>9.1f} {act_s['throughput']:>9.1f}")
    print()
    print(f"  D3QN vs Fixed:    queue {pct(fix_s['avg_queue'],  rl_s['avg_queue'])} "
          f" wait {pct(fix_s['avg_wait'],  rl_s['avg_wait'])}")
    print(f"  D3QN vs SOTL:     queue {pct(sotl_s['avg_queue'], rl_s['avg_queue'])} "
          f" wait {pct(sotl_s['avg_wait'], rl_s['avg_wait'])}")
    print(f"  D3QN vs Actuated: queue {pct(act_s['avg_queue'],  rl_s['avg_queue'])} "
          f" wait {pct(act_s['avg_wait'],  rl_s['avg_wait'])}")
    print("="*72)

    np.save("enhanced_eval_results.npy",
            {"rl": rl_s, "fixed": fix_s, "sotl": sotl_s, "actuated": act_s})
    print("\nResults saved → enhanced_eval_results.npy")


if __name__ == "__main__":
    main()
