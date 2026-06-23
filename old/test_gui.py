import os
import sys
import time

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

SUMO_GUI = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
CFG = "traffic.sumocfg"

os.chdir(WORKDIR)

traci.start([SUMO_GUI, "-c", CFG, "--start"])

for _ in range(1000):
    traci.simulationStep()
    time.sleep(0.05)

traci.close()
