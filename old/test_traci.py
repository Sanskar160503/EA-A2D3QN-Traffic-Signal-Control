import os
import sys
import time

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"

os.chdir(SUMO_WORKDIR)

sumo_cmd = [SUMO_BINARY, "-c", "traffic.sumocfg"]
traci.start(sumo_cmd)

TL_ID = "n_center"

print("Controlled lanes:")
print(traci.trafficlight.getControlledLanes(TL_ID))

for step in range(20):
    traci.simulationStep()

    # switch phase every 5 steps
    if step % 5 == 0:
        current = traci.trafficlight.getPhase(TL_ID)

        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(TL_ID)
        num_phases = len(program[0].phases)

        new_phase = (current + 1) % num_phases
        traci.trafficlight.setPhase(TL_ID, new_phase)

    print(f"Step {step} | Phase: {traci.trafficlight.getPhase(TL_ID)}")
    time.sleep(0.5)



traci.close()
