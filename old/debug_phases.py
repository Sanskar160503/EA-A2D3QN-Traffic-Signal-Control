# Run this once to print your exact SUMO phase definitions
# so we can fix the emergency phase mapping

import os, sys
SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci

SUMO_BINARY  = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG     = "traffic.sumocfg"
TL_ID        = "n_center"

LANES = [
    "north_to_center_0",
    "south_to_center_0",
    "east_to_center_0",
    "west_to_center_0"
]

os.chdir(SUMO_WORKDIR)
traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--start", "--quit-on-end"])
traci.simulationStep()

logic  = traci.trafficlight.getCompleteRedYellowGreenDefinition(TL_ID)
phases = logic[0].phases

print(f"\nTraffic light: {TL_ID}")
print(f"Number of phases: {len(phases)}\n")

for i, phase in enumerate(phases):
    state = phase.state
    print(f"Phase {i}: '{state}'  (duration={phase.duration}s)")
    for j, lane in enumerate(LANES):
        ch = state[j] if j < len(state) else '?'
        green = "GREEN" if ch.upper() == 'G' else "red  "
        print(f"         [{green}] lane index {j} = {lane.split('_')[0]}")
    print()

# Also print what lane indices each lane gets
print("Lane index mapping:")
for i, lane in enumerate(LANES):
    print(f"  index {i} → {lane}")

traci.close()
print("\nCopy the output above and share it.")
