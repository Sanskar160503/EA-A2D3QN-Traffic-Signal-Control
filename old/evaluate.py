# evaluate.py — Final evaluation for D3QN vs Fixed-time baseline
# Uses environment_v2.py (11-dim state) and dqn_final.pth

import torch
import numpy as np
import traci
from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv    # <-- updated from environment.py

SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
MODEL_PATH      = "dqn_final.pth"       # <-- updated from dqn_parallel.pth
EVAL_EPISODES   = 10
MAX_STEPS       = 1000


def run_episode(env, agent=None, fixed_time=False, fixed_interval=30):
    state = env.reset(use_gui=False)
    metrics = {
        "queue_lengths":   [],
        "wait_times":      [],
        "vehicles_passed": 0,
        "emergency_times": [],
    }
    emergency_start = None

    for step in range(MAX_STEPS):
        if fixed_time:
            action = 1 if step % fixed_interval == 0 else 0
        else:
            action = agent.act(state)

        next_state, reward, done = env.step(action)

        # Queue: state[:4] are normalized — multiply by 10 to get raw vehicle counts
        queues_raw = next_state[:4] * 10.0
        metrics["queue_lengths"].append(float(np.sum(queues_raw)))

        # Wait time: sum of halting vehicles this step
        wait = sum(traci.lane.getLastStepHaltingNumber(l) for l in env.lanes)
        metrics["wait_times"].append(float(wait))

        # Throughput
        metrics["vehicles_passed"] += traci.simulation.getArrivedNumber()

        # Emergency clearance tracking
        em_lane = env.detect_emergency()
        if em_lane and emergency_start is None:
            emergency_start = step
        elif not em_lane and emergency_start is not None:
            metrics["emergency_times"].append(step - emergency_start)
            emergency_start = None

        state = next_state
        if done:
            break

    return metrics


def evaluate():
    env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)

    # ── RL Agent ──
    agent = D3QNAgent(state_dim=11, action_dim=2)   # <-- 11-dim state
    agent.policy_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device, weights_only=True)
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0   # pure greedy — no exploration

    rl_results    = []
    fixed_results = []

    print(f"\n{'='*60}")
    print(f"  Evaluating: {EVAL_EPISODES} episodes | Model: {MODEL_PATH}")
    print(f"{'='*60}\n")

    for ep in range(EVAL_EPISODES):
        # Run RL episode
        rl_m = run_episode(env, agent=agent, fixed_time=False)
        # Run fixed-time episode  
        fixed_m = run_episode(env, agent=None, fixed_time=True, fixed_interval=30)

        rl_results.append(rl_m)
        fixed_results.append(fixed_m)

        print(
            f"  Ep {ep+1:2d} | "
            f"RL  queue={np.mean(rl_m['queue_lengths']):5.2f}  "
            f"wait={np.mean(rl_m['wait_times']):5.2f}  "
            f"passed={rl_m['vehicles_passed']:4d}  "
            f"emerg={np.mean(rl_m['emergency_times']) if rl_m['emergency_times'] else 0:.1f} | "
            f"Fixed queue={np.mean(fixed_m['queue_lengths']):5.2f}  "
            f"wait={np.mean(fixed_m['wait_times']):5.2f}  "
            f"passed={fixed_m['vehicles_passed']:4d}"
        )

    env.close()

    def summarize(results):
        emerg_avgs = [
            np.mean(r["emergency_times"]) if r["emergency_times"] else None
            for r in results
        ]
        valid_emerg = [x for x in emerg_avgs if x is not None]
        return {
            "avg_queue":        np.mean([np.mean(r["queue_lengths"]) for r in results]),
            "avg_wait":         np.mean([np.mean(r["wait_times"])    for r in results]),
            "avg_throughput":   np.mean([r["vehicles_passed"]        for r in results]),
            "avg_emerg_clear":  np.mean(valid_emerg) if valid_emerg else 0.0,
            "emerg_episodes":   len(valid_emerg),
        }

    rl_s    = summarize(rl_results)
    fixed_s = summarize(fixed_results)

    def pct(baseline, new):
        if baseline == 0:
            return 0.0
        return (baseline - new) / baseline * 100

    print(f"\n{'='*62}")
    print(f"  {'Metric':<28} {'RL Agent':>10} {'Fixed-time':>10} {'Improvement':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Avg queue length (veh)':<28} {rl_s['avg_queue']:>10.2f} {fixed_s['avg_queue']:>10.2f} {pct(fixed_s['avg_queue'], rl_s['avg_queue']):>+9.1f}%")
    print(f"  {'Avg wait (veh/step)':<28} {rl_s['avg_wait']:>10.2f} {fixed_s['avg_wait']:>10.2f} {pct(fixed_s['avg_wait'], rl_s['avg_wait']):>+9.1f}%")
    print(f"  {'Throughput (veh/episode)':<28} {rl_s['avg_throughput']:>10.1f} {fixed_s['avg_throughput']:>10.1f} {-pct(fixed_s['avg_throughput'], rl_s['avg_throughput']):>+9.1f}%")
    print(f"  {'Emerg. clear (steps)':<28} {rl_s['avg_emerg_clear']:>10.1f} {fixed_s['avg_emerg_clear']:>10.1f} {pct(fixed_s['avg_emerg_clear'], rl_s['avg_emerg_clear']):>+9.1f}%")
    print(f"  {'(episodes with emergency)':<28} {rl_s['emerg_episodes']:>10}  {fixed_s['emerg_episodes']:>10}")
    print(f"{'='*62}")

    # Save for paper
    np.save("eval_results_final.npy", {"rl": rl_s, "fixed": fixed_s})
    print(f"\nResults saved → eval_results_final.npy")
    print("Copy the table above into your paper.")


if __name__ == "__main__":
    evaluate()
