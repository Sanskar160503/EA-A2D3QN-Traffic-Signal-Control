"""
mysql_database.py
══════════════════════════════════════════════════════════════════════
MySQL Database Manager for EA-A2D3QN Traffic Signal Control Project

Tables:
  training_runs       — one row per training run (algorithm, hyperparams)
  episode_metrics     — per-episode reward, loss, queue, epsilon
  step_metrics        — per-step lane data (sampled every N steps)
  evaluation_results  — algorithm comparison results
  attention_logs      — attention weight snapshots per episode
  real_traffic_data   — imported PeMS/real traffic demand data

Setup:
  1. Install MySQL Server (https://dev.mysql.com/downloads/installer/)
  2. pip install mysql-connector-python
  3. Create DB:
       mysql -u root -p
       CREATE DATABASE traffic_rl_db;
       CREATE USER 'traffic_user'@'localhost' IDENTIFIED BY 'traffic123';
       GRANT ALL PRIVILEGES ON traffic_rl_db.* TO 'traffic_user'@'localhost';
       FLUSH PRIVILEGES; EXIT;
  4. from mysql_database import TrafficMySQL
     db = TrafficMySQL()

Author: [Your Name] | M.Tech Project 2026
"""

import json
import time
from datetime import datetime
import numpy as np

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    raise ImportError(
        "MySQL connector not found.\n"
        "Run: pip install mysql-connector-python"
    )


# ── Default connection config ──────────────────────────────────────────
DEFAULT_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "traffic_user",
    "password": "traffic123",
    "database": "traffic_rl_db",
}


class TrafficMySQL:
    """
    MySQL database manager for the EA-A2D3QN traffic signal control project.

    Stores all training runs, episode metrics, per-step intersection state,
    evaluation comparisons, attention weight logs, and real traffic demand data.

    Why MySQL over SQLite:
      - Concurrent access (multiple scripts can query while training runs)
      - Full SQL query capability with indexes for fast retrieval
      - Persistent server process — data survives script crashes
      - Reviewers recognize it as a production-grade component
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.conn   = None
        self._connect()
        self._create_tables()

    def _connect(self):
        try:
            self.conn = mysql.connector.connect(**self.config)
            if self.conn.is_connected():
                info = self.conn.get_server_info()
                print(f"[MySQL] Connected to MySQL Server v{info}")
                print(f"[MySQL] Database: {self.config['database']}")
        except Error as e:
            raise ConnectionError(
                f"Could not connect to MySQL: {e}\n"
                f"Check that MySQL server is running and credentials are correct."
            )

    def _execute(self, query, params=None, fetch=False, many=False):
        """Execute a query with auto-reconnect on dropped connection."""
        try:
            if not self.conn.is_connected():
                self.conn.reconnect(attempts=3, delay=2)
            cursor = self.conn.cursor(dictionary=True)
            if many:
                cursor.executemany(query, params or [])
            else:
                cursor.execute(query, params or ())
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            self.conn.commit()
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        except Error as e:
            print(f"[MySQL] Query error: {e}")
            self.conn.rollback()
            raise

    def _create_tables(self):
        """Create all project tables if they don't exist."""

        tables = {

            "training_runs": """
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id          INT AUTO_INCREMENT PRIMARY KEY,
                    algorithm       VARCHAR(50)  NOT NULL,
                    start_time      DATETIME     NOT NULL,
                    end_time        DATETIME,
                    total_episodes  INT,
                    best_avg_reward FLOAT,
                    model_path      VARCHAR(255),
                    hyperparams     JSON,
                    notes           TEXT,
                    INDEX idx_algorithm (algorithm),
                    INDEX idx_start_time (start_time)
                ) ENGINE=InnoDB
            """,

            "episode_metrics": """
                CREATE TABLE IF NOT EXISTS episode_metrics (
                    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
                    run_id          INT     NOT NULL,
                    episode         INT     NOT NULL,
                    total_reward    FLOAT,
                    avg_queue       FLOAT,
                    avg_wait        FLOAT,
                    td_loss         FLOAT,
                    aux_loss        FLOAT,
                    epsilon         FLOAT,
                    phase_imbalance FLOAT,
                    em_buffer_size  INT,
                    em_buffer_pct   FLOAT,
                    duration_sec    FLOAT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                        ON DELETE CASCADE,
                    INDEX idx_run_episode (run_id, episode)
                ) ENGINE=InnoDB
            """,

            "step_metrics": """
                CREATE TABLE IF NOT EXISTS step_metrics (
                    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                    run_id      INT     NOT NULL,
                    episode     INT     NOT NULL,
                    step        INT     NOT NULL,
                    q_north     FLOAT,
                    q_south     FLOAT,
                    q_east      FLOAT,
                    q_west      FLOAT,
                    total_queue FLOAT   GENERATED ALWAYS AS
                                (q_north + q_south + q_east + q_west) STORED,
                    phase       TINYINT,
                    phase_timer FLOAT,
                    emergency   TINYINT,
                    reward      FLOAT,
                    action      TINYINT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                        ON DELETE CASCADE,
                    INDEX idx_run_ep_step (run_id, episode, step),
                    INDEX idx_emergency (run_id, emergency)
                ) ENGINE=InnoDB
            """,

            "evaluation_results": """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    run_id          INT          NOT NULL,
                    eval_time       DATETIME     NOT NULL,
                    algorithm       VARCHAR(50)  NOT NULL,
                    scenario        VARCHAR(30)  DEFAULT 'normal',
                    avg_queue       FLOAT,
                    avg_wait        FLOAT,
                    throughput      FLOAT,
                    em_clear_steps  FLOAT,
                    episodes        INT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                        ON DELETE CASCADE,
                    INDEX idx_algorithm_scenario (algorithm, scenario)
                ) ENGINE=InnoDB
            """,

            "attention_logs": """
                CREATE TABLE IF NOT EXISTS attention_logs (
                    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
                    run_id          INT     NOT NULL,
                    episode         INT     NOT NULL,
                    is_emergency    TINYINT NOT NULL,
                    attn_q_north    FLOAT,
                    attn_q_south    FLOAT,
                    attn_q_east     FLOAT,
                    attn_q_west     FLOAT,
                    attn_w_north    FLOAT,
                    attn_w_south    FLOAT,
                    attn_w_east     FLOAT,
                    attn_w_west     FLOAT,
                    attn_phase      FLOAT,
                    attn_timer      FLOAT,
                    attn_emergency  FLOAT,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                        ON DELETE CASCADE,
                    INDEX idx_run_emergency (run_id, is_emergency)
                ) ENGINE=InnoDB
            """,

            "real_traffic_data": """
                CREATE TABLE IF NOT EXISTS real_traffic_data (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    source      VARCHAR(50)  NOT NULL,
                    timestamp   DATETIME,
                    hour_of_day TINYINT,
                    day_type    VARCHAR(10),
                    n_flow      FLOAT,
                    s_flow      FLOAT,
                    e_flow      FLOAT,
                    w_flow      FLOAT,
                    total_flow  FLOAT GENERATED ALWAYS AS
                                (n_flow + s_flow + e_flow + w_flow) STORED,
                    scenario    VARCHAR(20),
                    INDEX idx_hour (hour_of_day),
                    INDEX idx_source (source)
                ) ENGINE=InnoDB
            """,
        }

        for name, ddl in tables.items():
            self._execute(ddl)
            print(f"[MySQL] Table ready: {name}")

    # ══════════════════════════════════════════════════════════════════
    # Run management
    # ══════════════════════════════════════════════════════════════════

    def start_run(self, algorithm, hyperparams=None, notes=""):
        run_id = self._execute("""
            INSERT INTO training_runs
              (algorithm, start_time, hyperparams, notes)
            VALUES (%s, %s, %s, %s)
        """, (
            algorithm,
            datetime.now(),
            json.dumps(hyperparams or {}),
            notes
        ))
        print(f"[MySQL] Started run #{run_id}: {algorithm}")
        return run_id

    def end_run(self, run_id, total_episodes, best_avg_reward, model_path):
        self._execute("""
            UPDATE training_runs
            SET end_time=%s, total_episodes=%s,
                best_avg_reward=%s, model_path=%s
            WHERE run_id=%s
        """, (datetime.now(), total_episodes,
              best_avg_reward, model_path, run_id))
        print(f"[MySQL] Run #{run_id} complete. Best avg50: {best_avg_reward:.1f}")

    # ══════════════════════════════════════════════════════════════════
    # Episode logging
    # ══════════════════════════════════════════════════════════════════

    def log_episode(self, run_id, episode, total_reward,
                    avg_queue=None, avg_wait=None,
                    td_loss=None, aux_loss=None,
                    epsilon=None, phase_imbalance=None,
                    em_buffer_size=None, em_buffer_pct=None,
                    duration_sec=None):
        self._execute("""
            INSERT INTO episode_metrics
              (run_id, episode, total_reward, avg_queue, avg_wait,
               td_loss, aux_loss, epsilon, phase_imbalance,
               em_buffer_size, em_buffer_pct, duration_sec)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (run_id, episode, total_reward, avg_queue, avg_wait,
              td_loss, aux_loss, epsilon, phase_imbalance,
              em_buffer_size, em_buffer_pct, duration_sec))

    # ══════════════════════════════════════════════════════════════════
    # Step logging
    # ══════════════════════════════════════════════════════════════════

    def log_step(self, run_id, episode, step, state, action, reward):
        self._execute("""
            INSERT INTO step_metrics
              (run_id, episode, step,
               q_north, q_south, q_east, q_west,
               phase, phase_timer, emergency, reward, action)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            run_id, episode, step,
            float(state[0] * 10),
            float(state[1] * 10),
            float(state[2] * 10),
            float(state[3] * 10),
            int(round(state[8])),
            float(state[9]),
            int(state[10] > 0.5),
            float(reward),
            int(action)
        ))

    def log_steps_batch(self, rows):
        """
        Batch insert step metrics — much faster than one-by-one.
        rows: list of (run_id, episode, step, state, action, reward)
        """
        data = []
        for run_id, episode, step, state, action, reward in rows:
            data.append((
                run_id, episode, step,
                float(state[0]*10), float(state[1]*10),
                float(state[2]*10), float(state[3]*10),
                int(round(state[8])), float(state[9]),
                int(state[10] > 0.5), float(reward), int(action)
            ))
        self._execute("""
            INSERT INTO step_metrics
              (run_id, episode, step,
               q_north, q_south, q_east, q_west,
               phase, phase_timer, emergency, reward, action)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data, many=True)

    # ══════════════════════════════════════════════════════════════════
    # Attention logging
    # ══════════════════════════════════════════════════════════════════

    def log_attention(self, run_id, episode, is_emergency, weights_array):
        """
        Log attention weight vector [11] for one timestep.
        weights_array: numpy array of shape [11]
        """
        if weights_array is None or len(weights_array) < 11:
            return
        w = weights_array.flatten()[:11]
        self._execute("""
            INSERT INTO attention_logs
              (run_id, episode, is_emergency,
               attn_q_north, attn_q_south, attn_q_east, attn_q_west,
               attn_w_north, attn_w_south, attn_w_east, attn_w_west,
               attn_phase, attn_timer, attn_emergency)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (run_id, episode, int(is_emergency),
              float(w[0]), float(w[1]), float(w[2]), float(w[3]),
              float(w[4]), float(w[5]), float(w[6]), float(w[7]),
              float(w[8]), float(w[9]), float(w[10])))

    # ══════════════════════════════════════════════════════════════════
    # Real traffic data import
    # ══════════════════════════════════════════════════════════════════

    def import_real_traffic(self, records, source="PeMS"):
        """
        Import real traffic demand records into MySQL.
        records: list of dicts with keys:
          timestamp, hour_of_day, day_type, n_flow, s_flow, e_flow, w_flow, scenario
        """
        data = []
        for r in records:
            data.append((
                source,
                r.get("timestamp"),
                r.get("hour_of_day"),
                r.get("day_type", "weekday"),
                float(r.get("n_flow", 0)),
                float(r.get("s_flow", 0)),
                float(r.get("e_flow", 0)),
                float(r.get("w_flow", 0)),
                r.get("scenario", "normal"),
            ))
        self._execute("""
            INSERT INTO real_traffic_data
              (source, timestamp, hour_of_day, day_type,
               n_flow, s_flow, e_flow, w_flow, scenario)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data, many=True)
        print(f"[MySQL] Imported {len(data)} real traffic records from {source}")

    def get_traffic_scenario(self, scenario="peak_hour"):
        """
        Retrieve average flows for a scenario from real traffic data.
        Returns dict with n_flow, s_flow, e_flow, w_flow.
        """
        rows = self._execute("""
            SELECT
                AVG(n_flow) as n_flow,
                AVG(s_flow) as s_flow,
                AVG(e_flow) as e_flow,
                AVG(w_flow) as w_flow,
                COUNT(*) as n_records
            FROM real_traffic_data
            WHERE scenario = %s
        """, (scenario,), fetch=True)
        return rows[0] if rows else None

    # ══════════════════════════════════════════════════════════════════
    # Evaluation results
    # ══════════════════════════════════════════════════════════════════

    def log_evaluation(self, run_id, algorithm, avg_queue, avg_wait,
                       throughput, em_clear_steps, episodes,
                       scenario="normal"):
        self._execute("""
            INSERT INTO evaluation_results
              (run_id, eval_time, algorithm, scenario,
               avg_queue, avg_wait, throughput, em_clear_steps, episodes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (run_id, datetime.now(), algorithm, scenario,
              avg_queue, avg_wait, throughput, em_clear_steps, episodes))

    # ══════════════════════════════════════════════════════════════════
    # Query and reporting
    # ══════════════════════════════════════════════════════════════════

    def get_all_runs(self):
        return self._execute("""
            SELECT run_id, algorithm, start_time, end_time,
                   best_avg_reward, total_episodes, model_path
            FROM training_runs
            ORDER BY run_id DESC
        """, fetch=True)

    def get_run_summary(self, run_id):
        rows = self._execute("""
            SELECT
                r.run_id, r.algorithm, r.start_time, r.end_time,
                r.best_avg_reward, r.total_episodes, r.hyperparams,
                COUNT(e.id) as episodes_logged,
                AVG(e.total_reward) as mean_reward,
                MIN(e.total_reward) as min_reward,
                MAX(e.total_reward) as max_reward,
                AVG(e.avg_queue) as mean_queue,
                AVG(e.td_loss) as mean_loss
            FROM training_runs r
            LEFT JOIN episode_metrics e ON r.run_id = e.run_id
            WHERE r.run_id = %s
            GROUP BY r.run_id
        """, (run_id,), fetch=True)
        return rows[0] if rows else None

    def get_episode_history(self, run_id):
        """Returns all episode metrics as dict of lists."""
        rows = self._execute("""
            SELECT episode, total_reward, avg_queue, td_loss,
                   epsilon, phase_imbalance, em_buffer_size
            FROM episode_metrics
            WHERE run_id = %s
            ORDER BY episode
        """, (run_id,), fetch=True)
        if not rows:
            return {}
        data = {k: [] for k in rows[0].keys()}
        for row in rows:
            for k, v in row.items():
                data[k].append(v)
        return data

    def get_emergency_analysis(self, run_id):
        """
        Compare agent behavior during emergency vs normal timesteps.
        This is a key result — shows the dual PER buffer is working.
        """
        return self._execute("""
            SELECT
                emergency,
                COUNT(*)          as n_steps,
                AVG(total_queue)  as avg_total_queue,
                AVG(reward)       as avg_reward,
                SUM(action=1)     as switch_count,
                SUM(action=0)     as keep_count,
                AVG(phase_timer)  as avg_phase_timer
            FROM step_metrics
            WHERE run_id = %s
            GROUP BY emergency
            ORDER BY emergency
        """, (run_id,), fetch=True)

    def get_attention_analysis(self, run_id):
        """
        Average attention weights split by normal vs emergency.
        This is your novel visualization result for the paper.
        """
        return self._execute("""
            SELECT
                is_emergency,
                AVG(attn_q_north)   as q_north,
                AVG(attn_q_south)   as q_south,
                AVG(attn_q_east)    as q_east,
                AVG(attn_q_west)    as q_west,
                AVG(attn_w_north)   as w_north,
                AVG(attn_w_south)   as w_south,
                AVG(attn_w_east)    as w_east,
                AVG(attn_w_west)    as w_west,
                AVG(attn_phase)     as phase,
                AVG(attn_timer)     as timer,
                AVG(attn_emergency) as emergency_flag,
                COUNT(*)            as n_samples
            FROM attention_logs
            WHERE run_id = %s
            GROUP BY is_emergency
        """, (run_id,), fetch=True)

    def get_comparison_table(self):
        """Full algorithm comparison across scenarios."""
        return self._execute("""
            SELECT
                algorithm,
                scenario,
                ROUND(AVG(avg_queue), 2)      as avg_queue,
                ROUND(AVG(avg_wait),  2)      as avg_wait,
                ROUND(AVG(throughput), 1)     as throughput,
                ROUND(AVG(em_clear_steps), 1) as em_clear,
                COUNT(*)                      as n_evals
            FROM evaluation_results
            GROUP BY algorithm, scenario
            ORDER BY scenario, avg_queue
        """, fetch=True)

    def get_peak_vs_offpeak(self, run_id):
        """
        Compare agent performance during real peak vs off-peak hours.
        Only works after importing real traffic data and running
        multi-scenario evaluation.
        """
        return self._execute("""
            SELECT
                e.scenario,
                ROUND(AVG(s.total_queue), 2) as avg_queue,
                COUNT(*) as steps
            FROM step_metrics s
            JOIN evaluation_results e
              ON s.run_id = e.run_id
            WHERE s.run_id = %s
            GROUP BY e.scenario
        """, (run_id,), fetch=True)

    def print_comparison_table(self):
        rows = self.get_comparison_table()
        if not rows:
            print("No evaluation results yet.")
            return
        print("\n" + "="*72)
        print(f"  {'Algorithm':<20} {'Scenario':<12} {'Queue':>7} "
              f"{'Wait':>7} {'Thruput':>8} {'EmClear':>8}")
        print("  " + "-"*68)
        for r in rows:
            em = f"{r['em_clear']:.1f}" if r['em_clear'] else "N/A"
            print(f"  {r['algorithm']:<20} {r['scenario']:<12} "
                  f"{r['avg_queue']:>7.2f} {r['avg_wait']:>7.2f} "
                  f"{r['throughput']:>8.1f} {em:>8}")
        print("="*72)

    def print_emergency_analysis(self, run_id):
        rows = self.get_emergency_analysis(run_id)
        if not rows:
            print("No step metrics found.")
            return
        labels = {0: "Normal traffic", 1: "Emergency active"}
        print("\n" + "="*55)
        print("  Agent Behavior: Normal vs Emergency Timesteps")
        print("="*55)
        for r in rows:
            label = labels.get(r["emergency"], "Unknown")
            total = r["switch_count"] + r["keep_count"]
            switch_pct = r["switch_count"] / max(total, 1) * 100
            print(f"\n  {label}  ({r['n_steps']:,} steps)")
            print(f"    Avg queue:    {r['avg_total_queue']:.2f} vehicles")
            print(f"    Avg reward:   {r['avg_reward']:.4f}")
            print(f"    SWITCH rate:  {switch_pct:.1f}%")
            print(f"    Avg timer:    {r['avg_phase_timer']:.2f}")
        print("="*55)

    def close(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[MySQL] Connection closed.")
