import os, sys
sys.path.append(r"C:\Program Files\Eclipse\Sumo\tools")   # IMPORTANT

import traci

traci.start([r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe", "-c",
             r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files\traffic.sumocfg"])

print("GUI Views Available ->", traci.gui.getIDList())
input("Press ENTER to close...")
traci.close()
