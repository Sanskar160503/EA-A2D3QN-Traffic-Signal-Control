import os
import sys
import numpy as np
import random

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci


class TrafficEnv:
    def __init__(self, sumo_binary, sumo_gui_binary, sumo_cfg, sumo_workdir):
        self.sumo_binary = sumo_binary
        self.sumo_gui_binary = sumo_gui_binary
        self.sumo_cfg = sumo_cfg
        self.sumo_workdir = sumo_workdir
        self.tl_id = "n_center"

        self.lanes = [
            "north_to_center_0",
            "south_to_center_0",
            "east_to_center_0",
            "west_to_center_0"
        ]

        # Emergency lane → phase mapping
        self.emergency_phase_map = {
            "north_to_center_0": 0,
            "south_to_center_0": 2,
            "east_to_center_0": 0,
            "west_to_center_0": 2
        }

        # ✅ NEW: Minimum green time control
        self.phase_timer = 0
        self.min_green = 10   # minimum steps before switching

    # -------------------- Emergency Detection --------------------
    def detect_emergency(self):
        for lane in self.lanes:
            for v in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getTypeID(v) == "emergency":
                    return lane
        return None

    # -------------------- Random Emergency Injection --------------------
    def spawn_emergency(self, step):
        if random.random() < 0.005:
            route = random.choice([
                "emergency_ns",
                "emergency_sn",
                "emergency_ew",
                "emergency_we"
            ])
            vid = f"emergency_{step}"
            try:
                traci.vehicle.add(
                    vehID=vid,
                    routeID=route,
                    typeID="emergency",
                    departSpeed="max"
                )
                print("🚑 Emergency spawned:", vid, route)
            except:
                pass

    # -------------------- Start SUMO --------------------
    def start(self, use_gui=False):
        os.chdir(self.sumo_workdir)
        cmd = [
            self.sumo_gui_binary if use_gui else self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end"
        ]
        traci.start(cmd)

    # -------------------- Reset --------------------
    def reset(self, use_gui=False):
        if traci.isLoaded():
            traci.close()
        self.start(use_gui)

        # ✅ Reset timer
        self.phase_timer = 0

        traci.simulationStep()
        return self.get_state()

    # -------------------- State --------------------
    def get_state(self):
        queues = [traci.lane.getLastStepHaltingNumber(l) for l in self.lanes]
        phase = traci.trafficlight.getPhase(self.tl_id)
        emergency = 1 if self.detect_emergency() else 0
        return np.array(queues + [phase, emergency], dtype=np.float32)

    # -------------------- Step --------------------
    def step(self, action):
        emergency_lane = self.detect_emergency()
        emergency_active = False

        # ✅ Update phase timer
        self.phase_timer += 1

        # ---------------- Emergency Override ----------------
        if emergency_lane:
            phase = self.emergency_phase_map[emergency_lane]
            traci.trafficlight.setPhase(self.tl_id, phase)
            emergency_active = True

            # Reset timer to avoid immediate switching after emergency
            self.phase_timer = 0

        else:
            # ---------------- RL-controlled signal ----------------
            if action == 1 and self.phase_timer >= self.min_green:
                current = traci.trafficlight.getPhase(self.tl_id)
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)

                traci.trafficlight.setPhase(
                    self.tl_id,
                    (current + 1) % len(logic[0].phases)
                )

                # Reset timer after switching
                self.phase_timer = 0

        # ---------------- Simulation Step ----------------
        traci.simulationStep()

        # Spawn emergency
        self.spawn_emergency(traci.simulation.getTime())

        # ---------------- Next State ----------------
        next_state = self.get_state()

        # ---------------- ✅ Normalized Reward ----------------
        queues = next_state[:4]
        total_q = sum(queues)
        max_possible = 20 * 4   # tune if needed

        # Normalized queue penalty [-1, 0]
        queue_penalty = -(total_q / max_possible)

        # Emergency bonus [0, 1]
        emergency_bonus = 1.0 if emergency_active else 0.0

        reward = queue_penalty + emergency_bonus

        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done

    # -------------------- Close --------------------
    def close(self):
        if traci.isLoaded():
            traci.close()