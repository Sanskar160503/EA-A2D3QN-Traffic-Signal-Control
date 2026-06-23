"""
query_results.py
══════════════════════════════════════════════════════════════════════
Query and display all results from MySQL database.
Run this at any time during or after training to inspect results.

Usage:
    python query_results.py              # show all runs
    python query_results.py 1            # detailed report for run_id=1
    python query_results.py comparison   # show algorithm comparison table
"""

import sys
import numpy as np
from mysql_database import TrafficMySQL


def show_all_runs(db):
    runs = db.get_all_runs()
    if not runs:
        print("No training runs in database yet.")
        return
    print("\n" + "="*75)
    print(f"  {'ID':>3}  {'Algorithm':<20} {'Started':<20} "
          f"{'Episodes':>8} {'Best R':>8}")
    print("  " + "-"*71)
    for r in runs:
        ep  = r["total_episodes"] or "--"
        br  = f"{r['best_avg_reward']:.1f}" if r["best_avg_reward"] else "--"
        end = "running" if not r["end_time"] else str(r["end_time"])[:16]
        print(f"  {r['run_id']:>3}  {r['algorithm']:<20} "
              f"{str(r['start_time'])[:16]:<20} {ep:>8} {br:>8}")
    print("="*75)


def show_run_detail(db, run_id):
    summary = db.get_run_summary(run_id)
    if not summary:
        print(f"Run #{run_id} not found.")
        return

    print(f"\n{'='*60}")
    print(f"  Run #{run_id}: {summary['algorithm']}")
    print(f"{'='*60}")
    print(f"  Started:       {summary['start_time']}")
    print(f"  Ended:         {summary['end_time'] or 'Still running'}")
    print(f"  Episodes:      {summary['episodes_logged']}")
    print(f"  Mean reward:   {summary['mean_reward']:.1f}" if summary['mean_reward'] else "  Mean reward: N/A")
    print(f"  Best avg50:    {summary['best_avg_reward']:.1f}" if summary['best_avg_reward'] else "  Best avg50: N/A")
    print(f"  Mean queue:    {summary['mean_queue']:.2f}" if summary['mean_queue'] else "  Mean queue: N/A")
    print(f"  Mean TD loss:  {summary['mean_loss']:.5f}" if summary['mean_loss'] else "  Mean TD loss: N/A")

    # Emergency analysis
    print("\n  Emergency Analysis:")
    db.print_emergency_analysis(run_id)

    # Attention analysis
    print("\n  Attention Weight Analysis (from MySQL):")
    labels = ["q_N","q_S","q_E","q_W","w_N","w_S","w_E","w_W",
              "phase","timer","emerg"]
    attn_rows = db.get_attention_analysis(run_id)
    for row in attn_rows:
        label = "EMERGENCY" if row["is_emergency"] else "Normal  "
        vals  = [row[k] for k in
                 ["q_north","q_south","q_east","q_west",
                  "w_north","w_south","w_east","w_west",
                  "phase","timer","emergency_flag"]]
        top3 = sorted(zip(labels, vals), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{f}={v:.3f}" for f, v in top3)
        print(f"    {label} top features: {top_str}  ({row['n_samples']} samples)")

    # Real traffic data
    print("\n  Real Traffic Data (from MySQL):")
    for scenario in ["off_peak", "normal", "peak_hour"]:
        row = db.get_traffic_scenario(scenario)
        if row and row["n_records"]:
            print(f"    {scenario:<12}: N={row['n_flow']:.0f} "
                  f"S={row['s_flow']:.0f} "
                  f"E={row['e_flow']:.0f} "
                  f"W={row['w_flow']:.0f} veh/hr  "
                  f"({row['n_records']} records)")


def show_comparison(db):
    print("\n  Algorithm Comparison Table (from MySQL):")
    db.print_comparison_table()


def main():
    db = TrafficMySQL()

    if len(sys.argv) < 2:
        show_all_runs(db)
        print("\nUsage:")
        print("  python query_results.py           — list all runs")
        print("  python query_results.py <run_id>  — detailed report")
        print("  python query_results.py comparison — comparison table")

    elif sys.argv[1] == "comparison":
        show_comparison(db)

    else:
        try:
            run_id = int(sys.argv[1])
            show_run_detail(db, run_id)
        except ValueError:
            print(f"Unknown argument: {sys.argv[1]}")

    db.close()


if __name__ == "__main__":
    main()
