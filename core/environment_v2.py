import os, sys, numpy as np, random

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci


class TrafficEnv:
    def __init__(self, sumo_binary, sumo_gui_binary, sumo_cfg, sumo_workdir):
        self.sumo_binary     = sumo_binary
        self.sumo_gui_binary = sumo_gui_binary
        self.sumo_cfg        = sumo_cfg
        self.sumo_workdir    = sumo_workdir
        self.tl_id           = "n_center"

        self.lanes = [
            "north_to_center_0",
            "south_to_center_0",
            "east_to_center_0",
            "west_to_center_0"
        ]

        # Confirmed from debug_phases.py
        self.emergency_phase_map = {
            "north_to_center_0": 2,
            "south_to_center_0": 2,
            "east_to_center_0":  2,
            "west_to_center_0":  0,
        }

        self.phase_timer     = 0
        self.min_green       = 15
        self.max_phase_steps = 60
        self.prev_queues     = np.zeros(4)
        self.prev_passed     = 0
        self.n_phases        = None
        self._em_hold_timer  = 0
        self._em_active_lane = None
        self._em_hold_steps  = 20
        self._removed_vehs   = set()   # track removed vehicles to avoid double-remove

    def _clear_path_for_emergency(self, emergency_lane, emergency_vid):
        """Safely remove only regular vehicles ahead of the ambulance."""
        try:
            all_vehs = traci.lane.getLastStepVehicleIDs(emergency_lane)
        except:
            return

        try:
            em_pos = traci.vehicle.getLanePosition(emergency_vid)
        except:
            return

        # Get all currently active vehicle IDs to avoid removing already-gone vehicles
        try:
            active_ids = set(traci.vehicle.getIDList())
        except:
            return

        for v in all_vehs:
            if v == emergency_vid:
                continue
            if v in self._removed_vehs:
                continue
            if v not in active_ids:
                continue
            try:
                v_type = traci.vehicle.getTypeID(v)
                if v_type == "emergency":
                    continue
                v_pos = traci.vehicle.getLanePosition(v)
                # Only remove vehicles ahead of (closer to intersection than) ambulance
                if v_pos > em_pos:
                    traci.vehicle.remove(v)
                    self._removed_vehs.add(v)
            except:
                continue

    def detect_emergency(self):
        """Return (lane, vehicle_id) tuple or (None, None)."""
        try:
            active_ids = set(traci.vehicle.getIDList())
        except:
            return None, None

        for lane in self.lanes:
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
            except:
                continue
            for v in vehicles:
                if v not in active_ids:
                    continue
                try:
                    if traci.vehicle.getTypeID(v) == "emergency":
                        return lane, v
                except:
                    continue
        return None, None

    def spawn_emergency(self, step):
        if random.random() < 0.005:
            route = random.choice(["emergency_ns", "emergency_sn",
                                   "emergency_ew", "emergency_we"])
            vid = f"emergency_{step}"
            try:
                traci.vehicle.add(vehID=vid, routeID=route,
                                  typeID="emergency", departSpeed="max")
                print("Emergency spawned:", vid, route)
            except:
                pass

    def start(self, use_gui=False):
        os.chdir(self.sumo_workdir)
        cmd = [self.sumo_gui_binary if use_gui else self.sumo_binary,
               "-c", self.sumo_cfg, "--start", "--quit-on-end"]
        traci.start(cmd)

    def reset(self, use_gui=False):
        if traci.isLoaded():
            traci.close()
        self.start(use_gui)
        traci.simulationStep()
        self.phase_timer     = 0
        self.prev_queues     = np.zeros(4)
        self.prev_passed     = traci.simulation.getArrivedNumber()
        self._em_hold_timer  = 0
        self._em_active_lane = None
        self._removed_vehs   = set()
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)
        self.n_phases = len(logic[0].phases)
        return self.get_state()

    def get_state(self):
        queues = np.array([traci.lane.getLastStepHaltingNumber(l)
                           for l in self.lanes], dtype=np.float32)
        queues_norm      = np.clip(queues / 10.0, 0.0, 1.0)
        wait_times       = np.array([traci.lane.getWaitingTime(l)
                                      for l in self.lanes], dtype=np.float32)
        wait_norm        = np.clip(wait_times / 300.0, 0.0, 1.0)
        phase            = float(traci.trafficlight.getPhase(self.tl_id))
        phase_timer_norm = float(min(self.phase_timer / self.max_phase_steps, 1.0))
        em_lane, _       = self.detect_emergency()
        emergency        = 1.0 if em_lane else 0.0
        return np.array(list(queues_norm) + list(wait_norm) +
                        [phase, phase_timer_norm, emergency], dtype=np.float32)

    def step(self, action):
        em_lane, em_vid  = self.detect_emergency()
        emergency_active = False

        if em_lane and em_vid:
            self._clear_path_for_emergency(em_lane, em_vid)
            em_phase = self.emergency_phase_map[em_lane]
            traci.trafficlight.setPhase(self.tl_id, em_phase)
            try:
                traci.vehicle.setSpeed(em_vid, 15.0)
                traci.vehicle.setSpeedMode(em_vid, 32)
            except:
                pass
            self._em_active_lane = em_lane
            self._em_hold_timer  = self._em_hold_steps
            emergency_active     = True
            self.phase_timer     = 0

        elif self._em_hold_timer > 0:
            em_phase = self.emergency_phase_map.get(self._em_active_lane, 2)
            traci.trafficlight.setPhase(self.tl_id, em_phase)
            self._em_hold_timer -= 1
            emergency_active     = True
            self.phase_timer     = 0
            if self._em_hold_timer == 0:
                self._em_active_lane = None
                self._removed_vehs   = set()   # reset after clearance complete

        else:
            self._em_active_lane = None
            self.phase_timer    += 1
            should_switch = (
                (action == 1 and self.phase_timer >= self.min_green) or
                (self.phase_timer >= self.max_phase_steps)
            )
            if should_switch:
                current = traci.trafficlight.getPhase(self.tl_id)
                traci.trafficlight.setPhase(self.tl_id,
                                            (current + 1) % self.n_phases)
                self.phase_timer = 0

        # Wrap simulationStep in try/except so a SUMO crash is caught gracefully
        try:
            traci.simulationStep()
        except Exception as e:
            print(f"SUMO step error at phase cleanup: {e}")
            return self.get_state(), 0.0, True

        self.spawn_emergency(traci.simulation.getTime())

        try:
            next_state = self.get_state()
        except:
            return np.zeros(11, dtype=np.float32), 0.0, True

        raw_queues = np.array([traci.lane.getLastStepHaltingNumber(l)
                                for l in self.lanes], dtype=np.float32)

        avg_queue_norm    = float(np.mean(raw_queues)) / 10.0
        queue_penalty     = -avg_queue_norm * 0.3
        queue_delta       = float(np.sum(self.prev_queues) - np.sum(raw_queues))
        reduction_bonus   = np.clip(queue_delta / 4.0, -1.0, 1.0) * 0.2
        current_passed    = traci.simulation.getArrivedNumber()
        passed_this_step  = current_passed - self.prev_passed
        throughput_bonus  = float(np.clip(passed_this_step, 0, 3)) / 3.0 * 0.3
        self.prev_passed  = current_passed
        imbalance         = float(np.max(raw_queues) - np.min(raw_queues))
        imbalance_penalty = -np.clip(imbalance / 10.0, 0.0, 1.0) * 0.1
        emergency_bonus   = 0.5 if emergency_active else 0.0

        reward = (queue_penalty + reduction_bonus + throughput_bonus
                  + imbalance_penalty + emergency_bonus)

        self.prev_queues = raw_queues.copy()
        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done

    def close(self):
        if traci.isLoaded():
            traci.close()
