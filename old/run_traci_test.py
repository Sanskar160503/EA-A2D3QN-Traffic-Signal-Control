import os, sys, traci

SUMO_HOME = r"C:\Program Files\Eclipse\Sumo"
sys.path.append(os.path.join(SUMO_HOME, "tools"))

SUMO_GUI = r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_CFG = r"traffic.sumocfg"
SUMO_DIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"

print("\n=== Starting SUMO GUI Test via TraCI ===")

os.chdir(SUMO_DIR)

traci.start([
    SUMO_GUI,
    "-c", SUMO_CFG,
    "--start",
    "--quit-on-end=false",
    "--time-to-teleport=-1"
])

print("SUMO launched!")
step = 0
while step < 200:
    traci.simulationStep()
    step += 1

traci.close()
print("TraCI test finished OK")
