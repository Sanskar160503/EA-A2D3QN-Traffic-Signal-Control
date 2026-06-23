"""
evaluate_emissions.py
══════════════════════════════════════════════════════════════════════
CO2 emissions and fuel consumption evaluation.

SUMO can compute per-vehicle emissions using the HBEFA3 model.
This adds a 4th evaluation metric to your comparison table:
  - Avg queue (existing)
  - Avg wait (existing)
  - Throughput (existing)
  - CO2 emissions in grams per episode (NEW)

Why this matters:
  - Directly connects your work to environmental sustainability
  - Smart cities agenda — reducing urban CO2 is a government priority
  - Stronger paper and patent angle: "reduces CO2 by X% vs fixed-time"

Usage:
    python evaluate_emissions.py
"""

import torch
import numpy as np
import traci
import os
from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MODEL_PATH      = "ea_a2d3qn_final.pth"
EVAL_EPISODES   = 5
MAX_STEPS       = 1000


def run_episode_with_emissions(env, agent=None, fixed_time=False):
    """
    Run episode and track CO2 emissions and fuel consumption.
    SUMO computes these using the HBEFA3 emission model automatically.
    """
    state = env.reset(use_gui=False)
    metrics = {
        "queue":    [],
        "co2_mg":   0.0,    # total CO2 in milligrams
        "fuel_ml":  0.0,    # total fuel in millilitres
        "passed":   0,
        "wait_sum": 0.0,
    }

    for step in range(MAX_STEPS):
        if fixed_time:
            action = 1 if step % 30 == 0 else 0
        else:
            action = agent.act(state)

        next_state, reward, done = env.step(action)

        # Collect per-vehicle emissions this step
        for veh_id in traci.vehicle.getIDList():
            try:
                metrics["co2_mg"]  += traci.vehicle.getCO2Emission(veh_id)
                metrics["fuel_ml"] += traci.vehicle.getFuelConsumption(veh_id)
            except Exception:
                pass

        metrics["queue"].append(float(np.sum(next_state[:4]) * 10))
        metrics["passed"] += traci.simulation.getArrivedNumber()

        # Waiting time: vehicles stopped this step
        for lane in env.lanes:
            metrics["wait_sum"] += traci.lane.getLastStepHaltingNumber(lane)

        state = next_state
        if done:
            break

    # Convert mg CO2 → grams, ml fuel → litres
    metrics["co2_g"]  = metrics["co2_mg"]  / 1000.0
    metrics["fuel_l"] = metrics["fuel_ml"] / 1000.0
    return metrics


def main():
    # Try to load EA-A2D3QN first, fall back to D3QN
    try:
        from ea_a2d3qn_agent import EA_A2D3QNAgent
        agent = EA_A2D3QNAgent(state_dim=11, action_dim=2)
        agent.policy_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=agent.device, weights_only=True)
        )
        agent_name = "EA-A2D3QN"
    except Exception:
        agent = D3QNAgent(state_dim=11, action_dim=2)
        agent.policy_net.load_state_dict(
            torch.load("dqn_final.pth", map_location=agent.device, weights_only=True)
        )
        agent_name = "D3QN"

    agent.policy_net.eval()
    agent.epsilon = 0.0

    env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)

    print(f"\n{'='*60}")
    print(f"  Emissions Evaluation: {agent_name} vs Fixed-time")
    print(f"  Metric: CO2 (g/episode), Fuel (L/episode)")
    print(f"{'='*60}\n")

    rl_results    = []
    fixed_results = []

    print(f"[1/2] {agent_name} ({EVAL_EPISODES} episodes)...")
    for ep in range(EVAL_EPISODES):
        m = run_episode_with_emissions(env, agent=agent)
        rl_results.append(m)
        print(f"  Ep {ep+1}: queue={np.mean(m['queue']):.2f}  "
              f"CO2={m['co2_g']:.0f}g  fuel={m['fuel_l']:.2f}L")

    print(f"\n[2/2] Fixed-time ({EVAL_EPISODES} episodes)...")
    for ep in range(EVAL_EPISODES):
        m = run_episode_with_emissions(env, fixed_time=True)
        fixed_results.append(m)
        print(f"  Ep {ep+1}: queue={np.mean(m['queue']):.2f}  "
              f"CO2={m['co2_g']:.0f}g  fuel={m['fuel_l']:.2f}L")

    env.close()

    # Summarize
    def avg(results, key):
        return np.mean([r[key] for r in results])

    rl_q   = avg(rl_results,    "co2_g");   fx_q   = avg(fixed_results, "co2_g")
    rl_f   = avg(rl_results,    "fuel_l");  fx_f   = avg(fixed_results, "fuel_l")
    rl_que = avg(rl_results,    "queue");   fx_que = avg(fixed_results, "queue")
    rl_th  = avg(rl_results,    "passed");  fx_th  = avg(fixed_results, "passed")

    def pct(base, new):
        return f"{(base-new)/base*100:+.1f}%"

    print(f"\n{'='*62}")
    print(f"  {'Metric':<28} {agent_name:>12} {'Fixed':>8} {'Change':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Avg queue (veh)':<28} {rl_que:>12.2f} {fx_que:>8.2f} "
          f"{pct(fx_que, rl_que):>10}")
    print(f"  {'Throughput (veh/ep)':<28} {rl_th:>12.1f} {fx_th:>8.1f} "
          f"{pct(fx_th, rl_th):>+10}")
    print(f"  {'CO2 emissions (g/ep)':<28} {rl_q:>12.0f} {fx_q:>8.0f} "
          f"{pct(fx_q, rl_q):>10}")
    print(f"  {'Fuel consumption (L/ep)':<28} {rl_f:>12.2f} {fx_f:>8.2f} "
          f"{pct(fx_f, rl_f):>10}")
    print(f"{'='*62}")

    print(f"\nKey finding:")
    print(f"  {agent_name} reduces CO2 by {pct(fx_q, rl_q)} vs fixed-time")
    print(f"  Lower queue → less idling → less fuel burned → less CO2")
    print(f"\nFor your paper: cite SUMO's HBEFA3 emission model")
    print(f"  Ref: 'Handbook Emission Factors for Road Transport (HBEFA)'")
    print(f"  SUMO emission model: sumo.dlr.de/userdoc/Models/Emissions")

    # Save
    np.save("emissions_results.npy", {
        "rl_co2":   rl_q,  "fixed_co2":  fx_q,
        "rl_fuel":  rl_f,  "fixed_fuel": fx_f,
        "algorithm": agent_name,
    })
    print("\nSaved → emissions_results.npy")


if __name__ == "__main__":
    main()
