import os, sys, time
SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

SUMO_GUI = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
CFG = "traffic.sumocfg"

tl_id = "n_center"

os.chdir(WORKDIR)
traci.start([SUMO_GUI, "-c", CFG, "--start"])

step = 0
while step < 2000:
    traci.simulationStep()

    # Overlay text
    queues = sum([traci.lane.getLastStepHaltingNumber(l) for l in [
        "north_to_center_0","south_to_center_0","east_to_center_0","west_to_center_0"
    ]])
    phase = traci.trafficlight.getPhase(tl_id)

    text = f"Step: {step} | Phase: {phase} | Total Queue: {queues}"
    traci.gui.setText("View #0", text)

    time.sleep(0.25)  # slows GUI so overlay is visible
    step += 1

traci.close()
print("Overlay Test Finished")
