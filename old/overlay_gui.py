import os, sys, time

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci

SUMO_GUI = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
CFG = "traffic.sumocfg"
tl_id = "n_center"

os.chdir(WORKDIR)

# 🚦 Run SUMO in GUI mode WITH required flags
cmd = [
    SUMO_GUI, "-c", CFG,
    "--start",
    "--quit-on-end", "false",        # ❗ prevents GUI from auto closing
    "--time-to-teleport", "-1",      # ❗ prevents vehicles leaving simulation
    "--delay", "200",                # slow GUI refresh
    "--step-length", "0.5"           # 2x slower steps
]

traci.start(cmd)

step = 0
while True:
    traci.simulationStep()

    queues = sum([
        traci.lane.getLastStepHaltingNumber(l)
        for l in ["north_to_center_0","south_to_center_0","east_to_center_0","west_to_center_0"]
    ])
    phase = traci.trafficlight.getPhase(tl_id)

    text = f"Step: {step} | Phase: {phase} | Queue Total: {queues}"
    traci.gui.setText("View #0", text)

    time.sleep(0.2)
    step += 1
