import os
import sys

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"

os.chdir(SUMO_WORKDIR)

traci.start([SUMO_BINARY, "-c", "traffic.sumocfg"])

print("Traffic Light IDs in this network:")
print(traci.trafficlight.getIDList())

traci.close()
