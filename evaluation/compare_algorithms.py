"""
compare_algorithms.py  (FIXED v2)
══════════════════════════════════════════════════════════════════════
Direct comparison: EA-A2D3QN vs D3QN (baseline) vs Fixed-time
 
Fixes applied vs original:
  FIX A: Emergency vehicles spawned with traci.route.add() (dynamic),
          same method as train_fixed.py — eliminates "Invalid route
          'emergency_ew'/'emergency_we'" errors.
  FIX B: Emergency detection uses traci.vehicle.getTypeID() directly,
          same method as train_fixed.py — fixes EmClear = NaN.
  FIX C: emergency_start / clearance timer now correctly measures steps
          from first detection to last detection (not first depart).
  FIX D: Emergency vehicles given right-of-way via setSpeedMode so
          they don't teleport/collide and disappear before detection.
 
Run AFTER training both models:
    python compare_algorithms.py
"""
 
import torch
import numpy as np
import random
import traci
 
from ea_a2d3qn_agent import EA_A2D3QNAgent
from agent_v3 import D3QNAgent
from environment_v2 import TrafficEnv
 
# ── Paths ──────────────────────────────────────────────────────────────
SUMO_BINARY     = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_GUI_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_WORKDIR    = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG        = "traffic.sumocfg"
EA_MODEL_PATH   = "ea_a2d3qn_final.pth"
D3QN_MODEL_PATH = "dqn_d3qn_v3.pth"
 
# ── Eval settings ──────────────────────────────────────────────────────
EVAL_EPISODES      = 10
MAX_STEPS          = 1000
EMERGENCY_TYPE_ID  = "emergency"
EMERGENCY_INTERVAL = 200   # spawn one every ~200 steps (guaranteed 4-5 per episode)
 
# FIX A: Correct edge names confirmed from debug_phases.py output:
#   Incoming: north_to_center, south_to_center, east_to_center, west_to_center
#   Outgoing: center_to_south, center_to_north, center_to_west, center_to_east
EMERGENCY_ROUTES = [
    ("north_to_center center_to_south", "north_to_center_0"),  # NS
    ("south_to_center center_to_north", "south_to_center_0"),  # SN
    ("east_to_center center_to_west",   "east_to_center_0"),   # EW
    ("west_to_center center_to_east",   "west_to_center_0"),   # WE
]
 
_em_type_initialized = False   # per-episode flag
 
 
def _init_emergency_type():
    """Define the emergency vehicle type once per SUMO session."""
    global _em_type_initialized
    if _em_type_initialized:
        return
    try:
        known = traci.vehicletype.getIDList()
        if EMERGENCY_TYPE_ID not in known:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", EMERGENCY_TYPE_ID)
            traci.vehicletype.setColor(EMERGENCY_TYPE_ID, (255, 0, 0, 255))
            traci.vehicletype.setMaxSpeed(EMERGENCY_TYPE_ID, 22.22)  # ~80 km/h
        _em_type_initialized = True
    except Exception as e:
        print(f"  [warn] Could not init emergency type: {e}")
 
 
def spawn_emergency(step, em_counter):
    """
    FIX A: Dynamically create route + vehicle via TraCI.
    Returns (spawned: bool, new_counter: int).
    """
    if step == 0 or step % EMERGENCY_INTERVAL != 0:
        return False, em_counter
 
    _init_emergency_type()
 
    edges, lane = random.choice(EMERGENCY_ROUTES)
    em_id    = f"em_{em_counter}"
    route_id = f"em_route_{em_counter}"
 
    try:
        traci.route.add(route_id, edges.split())
        traci.vehicle.add(
            em_id, route_id,
            typeID      = EMERGENCY_TYPE_ID,
            depart      = "now",
            departSpeed = "0",
            departLane  = "free",
        )
        # FIX D: give emergency vehicle right-of-way so it doesn't
        # immediately collide and teleport out before we detect it.
        # speedMode 7 = obey signals but ignore safe-speed checks vs others
        traci.vehicle.setSpeedMode(em_id, 7)
        traci.vehicle.setMaxSpeed(em_id, 22.22)
        return True, em_counter + 1
    except Exception as e:
        # Route or insertion failed — silently skip
        return False, em_counter
 
 
def detect_emergency_active():
    """
    FIX B: Check for any live vehicle whose type is EMERGENCY_TYPE_ID.
    Returns True if at least one emergency vehicle is currently in sim.
    """
    try:
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == EMERGENCY_TYPE_ID:
                return True
    except Exception:
        pass
    return False
 
 
def run_episode(env, agent=None, fixed_time=False):
    """
    Run one evaluation episode.
    Returns dict of per-episode metrics.
    """
    global _em_type_initialized
    _em_type_initialized = False   # reset so type gets re-registered each episode
 
    state = env.reset(use_gui=False)
 
    metrics = {
        "queue":    [],
        "wait":     [],
        "passed":   0,
        "em_times": [],   # clearance times in steps
        "em_steps": 0,    # total steps where emergency was active
    }
 
    em_counter    = 0
    em_active     = False
    em_start_step = None
 
    for step in range(MAX_STEPS):
        # Spawn emergency vehicle every EMERGENCY_INTERVAL steps
        spawned, em_counter = spawn_emergency(step, em_counter)
 
        # Agent selects action
        if fixed_time:
            action = 1 if step % 30 == 0 else 0
        else:
            action = agent.act(state)
 
        next_state, reward, done = env.step(action)
 
        # ── Queue & wait metrics ───────────────────────────────────────
        queues_raw = next_state[:4] * 10.0
        metrics["queue"].append(float(np.sum(queues_raw)))
 
        try:
            wait = sum(traci.lane.getLastStepHaltingNumber(l)
                       for l in env.lanes)
        except Exception:
            wait = float(np.sum(queues_raw))
        metrics["wait"].append(float(wait))
 
        try:
            metrics["passed"] += traci.simulation.getArrivedNumber()
        except Exception:
            pass
 
        # FIX B: Emergency detection via vehicle type
        currently_emergency = detect_emergency_active()
 
        if currently_emergency:
            metrics["em_steps"] += 1
            if not em_active:
                # Emergency just started
                em_active     = True
                em_start_step = step
        else:
            if em_active:
                # Emergency just cleared
                clearance_steps = step - em_start_step
                metrics["em_times"].append(clearance_steps)
                em_active = False
 
        state = next_state
        if done:
            break
 
    # If episode ends while emergency still active, count it
    if em_active and em_start_step is not None:
        metrics["em_times"].append(MAX_STEPS - em_start_step)
 
    return metrics
 
 
def summarize(results):
    valid_em = [np.mean(r["em_times"]) for r in results if r["em_times"]]
    return {
        "avg_queue":   np.mean([np.mean(r["queue"]) for r in results]),
        "avg_wait":    np.mean([np.mean(r["wait"])  for r in results]),
        "throughput":  np.mean([r["passed"]          for r in results]),
        "em_clear":    np.mean(valid_em) if valid_em else float("nan"),
        "em_episodes": sum(1 for r in results if r["em_times"]),
        "em_steps":    np.mean([r["em_steps"] for r in results]),
    }
 
 
def pct_improvement(base, new):
    """% improvement of new over base (positive = better)."""
    if base == 0 or base != base:   # 0 or NaN
        return "N/A"
    return f"{(base - new) / base * 100:+.1f}%"
 
 
def main():
    env = TrafficEnv(SUMO_BINARY, SUMO_GUI_BINARY, SUMO_CFG, SUMO_WORKDIR)
 
    # ── Load EA-A2D3QN ─────────────────────────────────────────────────
    ea_agent = EA_A2D3QNAgent(state_dim=11, action_dim=2)
    ea_agent.policy_net.load_state_dict(
        torch.load(EA_MODEL_PATH,
                   map_location=ea_agent.device,
                   weights_only=True)
    )
    ea_agent.policy_net.eval()
    ea_agent.epsilon = 0.0
 
    # ── Load D3QN ──────────────────────────────────────────────────────
    d3qn_agent = D3QNAgent(state_dim=11, action_dim=2)
    d3qn_agent.policy_net.load_state_dict(
        torch.load(D3QN_MODEL_PATH,
                   map_location=d3qn_agent.device,
                   weights_only=True)
    )
    d3qn_agent.policy_net.eval()
    d3qn_agent.epsilon = 0.0
 
    print(f"\n{'='*65}")
    print(f"  Algorithm Comparison (FIXED v2): EA-A2D3QN vs D3QN vs Fixed-time")
    print(f"  {EVAL_EPISODES} eval episodes | ε=0 | Emergency every {EMERGENCY_INTERVAL} steps")
    print(f"{'='*65}\n")
 
    # ── Evaluate ───────────────────────────────────────────────────────
    for label, agent, fixed, store in [
        ("EA-A2D3QN (novel)", ea_agent,   False, "ea"),
        ("D3QN (baseline)",   d3qn_agent, False, "d3"),
        ("Fixed-time",        None,       True,  "fx"),
    ]:
        print(f"\n[{store.upper()}] Evaluating {label}...")
        results = []
        for ep in range(EVAL_EPISODES):
            m = run_episode(env, agent=agent, fixed_time=fixed)
            results.append(m)
            em_str = (f"em_clear={np.mean(m['em_times']):.1f}steps"
                      if m["em_times"] else "em=none_detected")
            print(f"  Ep {ep+1:2d}: "
                  f"queue={np.mean(m['queue']):.2f}  "
                  f"wait={np.mean(m['wait']):.2f}  "
                  f"passed={m['passed']}  "
                  f"em_steps={m['em_steps']}  "
                  f"{em_str}")
        if store == "ea":
            ea_results = results
        elif store == "d3":
            d3_results = results
        else:
            fx_results = results
 
    env.close()
 
    ea_s = summarize(ea_results)
    d3_s = summarize(d3_results)
    fx_s = summarize(fx_results)
 
    # ── Print comparison table ─────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  {'Metric':<30} {'EA-A2D3QN':>12} {'D3QN':>10} {'Fixed':>10}")
    print(f"  {'-'*74}")
 
    print(f"  {'Avg queue length (veh)':<30} "
          f"{ea_s['avg_queue']:>12.2f} "
          f"{d3_s['avg_queue']:>10.2f} "
          f"{fx_s['avg_queue']:>10.2f}")
    print(f"  {'  vs Fixed-time':<30} "
          f"{pct_improvement(fx_s['avg_queue'], ea_s['avg_queue']):>12} "
          f"{pct_improvement(fx_s['avg_queue'], d3_s['avg_queue']):>10}")
    print(f"  {'  vs D3QN':<30} "
          f"{pct_improvement(d3_s['avg_queue'], ea_s['avg_queue']):>12}")
    print()
 
    print(f"  {'Avg wait (veh/step)':<30} "
          f"{ea_s['avg_wait']:>12.2f} "
          f"{d3_s['avg_wait']:>10.2f} "
          f"{fx_s['avg_wait']:>10.2f}")
    print(f"  {'  vs Fixed-time':<30} "
          f"{pct_improvement(fx_s['avg_wait'], ea_s['avg_wait']):>12} "
          f"{pct_improvement(fx_s['avg_wait'], d3_s['avg_wait']):>10}")
    print()
 
    print(f"  {'Throughput (veh/ep)':<30} "
          f"{ea_s['throughput']:>12.1f} "
          f"{d3_s['throughput']:>10.1f} "
          f"{fx_s['throughput']:>10.1f}")
    print()
 
    print(f"  {'Avg em steps/episode':<30} "
          f"{ea_s['em_steps']:>12.1f} "
          f"{d3_s['em_steps']:>10.1f} "
          f"{fx_s['em_steps']:>10.1f}")
    print()
 
    em_ea = f"{ea_s['em_clear']:.1f}" if ea_s['em_clear'] == ea_s['em_clear'] else "N/A"
    em_d3 = f"{d3_s['em_clear']:.1f}" if d3_s['em_clear'] == d3_s['em_clear'] else "N/A"
    em_fx = f"{fx_s['em_clear']:.1f}" if fx_s['em_clear'] == fx_s['em_clear'] else "N/A"
    print(f"  {'Emerg. clearance (steps)':<30} "
          f"{em_ea:>12} {em_d3:>10} {em_fx:>10}")
    print(f"  {'  (episodes with em events)':<30} "
          f"{ea_s['em_episodes']:>12} "
          f"{d3_s['em_episodes']:>10} "
          f"{fx_s['em_episodes']:>10}")
    print(f"{'='*78}")
 
    # ── Key improvements ───────────────────────────────────────────────
    print(f"\nKey improvement of EA-A2D3QN over D3QN:")
    print(f"  Queue:      {pct_improvement(d3_s['avg_queue'], ea_s['avg_queue'])}")
    print(f"  Wait:       {pct_improvement(d3_s['avg_wait'],  ea_s['avg_wait'])}")
    if ea_s['em_clear'] == ea_s['em_clear'] and d3_s['em_clear'] == d3_s['em_clear']:
        print(f"  Em.clear:   {pct_improvement(d3_s['em_clear'], ea_s['em_clear'])}")
    else:
        print(f"  Em.clear:   (insufficient em detection — see em_steps above)")
 
    np.save("comparison_results.npy", {"ea": ea_s, "d3qn": d3_s, "fixed": fx_s})
    print("\nResults saved → comparison_results.npy")
 
    # ── Log to MySQL ───────────────────────────────────────────────────
    try:
        from mysql_database import TrafficMySQL
        db = TrafficMySQL()
        run_id = 9   # adjust to your latest training run ID
        for algo, s in [("EA-A2D3QN", ea_s), ("D3QN", d3_s), ("Fixed-time", fx_s)]:
            em_val = s["em_clear"] if s["em_clear"] == s["em_clear"] else None
            db.log_evaluation(
                run_id         = run_id,
                algorithm      = algo,
                avg_queue      = s["avg_queue"],
                avg_wait       = s["avg_wait"],
                throughput     = s["throughput"],
                em_clear_steps = em_val,
                episodes       = EVAL_EPISODES,
                scenario       = "normal",
            )
        db.close()
        print("Results also saved → MySQL evaluation_results")
        print("Query: python query_results.py comparison")
    except Exception as e:
        print(f"MySQL log skipped: {e}")
 
    # ── Final clean summary ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  EA-A2D3QN avg queue:  {ea_s['avg_queue']:.2f} veh")
    print(f"  D3QN avg queue:       {d3_s['avg_queue']:.2f} veh")
    print(f"  Fixed-time queue:     {fx_s['avg_queue']:.2f} veh")
    print()
    print(f"  EA-A2D3QN vs Fixed:   "
          f"{(fx_s['avg_queue']-ea_s['avg_queue'])/fx_s['avg_queue']*100:.1f}% improvement")
    print(f"  EA-A2D3QN vs D3QN:    "
          f"{(d3_s['avg_queue']-ea_s['avg_queue'])/d3_s['avg_queue']*100:.1f}% improvement")
    print(f"  Throughput gain:      "
          f"{ea_s['throughput']-fx_s['throughput']:+.1f} veh/ep vs fixed-time")
    if ea_s['em_clear'] == ea_s['em_clear']:
        print(f"  EA-A2D3QN em clear:   {ea_s['em_clear']:.1f} steps")
        print(f"  Fixed-time em clear:  {fx_s['em_clear']:.1f} steps")
    print(f"{'='*55}")
 
 
if __name__ == "__main__":
    main()