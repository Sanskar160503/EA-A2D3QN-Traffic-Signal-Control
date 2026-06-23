"""
read_network.py
══════════════════════════════════════════════════════════════════════
Reads your actual SUMO network and prints:
  - All edge IDs
  - All existing route IDs
  - All vehicle type IDs
  - Traffic light IDs and phase strings

Run this FIRST before train_fixed.py to get your real edge names.

Usage:
    python read_network.py
"""

import os
import subprocess
import xml.etree.ElementTree as ET
import traci

SUMO_BINARY = r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"
SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG    = "traffic.sumocfg"


def read_from_xml():
    """Read network directly from XML files without starting SUMO."""
    print("\n" + "="*60)
    print("  Reading SUMO network XML files directly")
    print("="*60)

    # Find net.xml file
    net_file = None
    rou_file = None
    cfg_path = os.path.join(SUMO_WORKDIR, SUMO_CFG)

    if os.path.exists(cfg_path):
        tree = ET.parse(cfg_path)
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag == "net-file":
                net_file = os.path.join(SUMO_WORKDIR, elem.get("value", ""))
            if elem.tag == "route-files":
                rou_file = os.path.join(SUMO_WORKDIR, elem.get("value", ""))
        print(f"\n  Config: {cfg_path}")
        print(f"  Net file: {net_file}")
        print(f"  Route file: {rou_file}")

    # Read edges from net.xml
    if net_file and os.path.exists(net_file):
        print(f"\n  ── Edges in {os.path.basename(net_file)} ──")
        net_tree = ET.parse(net_file)
        net_root = net_tree.getroot()

        edges = []
        for edge in net_root.findall("edge"):
            eid = edge.get("id", "")
            if not eid.startswith(":"):   # skip internal junction edges
                efrom = edge.get("from", "")
                eto   = edge.get("to", "")
                edges.append((eid, efrom, eto))

        print(f"  Total non-internal edges: {len(edges)}")
        for eid, efrom, eto in sorted(edges):
            print(f"    '{eid}'  ({efrom} → {eto})")

        # Read junctions
        print(f"\n  ── Junctions ──")
        for junc in net_root.findall("junction"):
            jid   = junc.get("id", "")
            jtype = junc.get("type", "")
            if not jid.startswith(":"):
                print(f"    '{jid}'  type={jtype}")

        # Read traffic lights
        print(f"\n  ── Traffic Lights ──")
        for tl in net_root.findall("tlLogic"):
            tlid = tl.get("id", "")
            print(f"    TL ID: '{tlid}'")
            for i, phase in enumerate(tl.findall("phase")):
                state = phase.get("state", "")
                dur   = phase.get("duration", "")
                print(f"      Phase {i}: '{state}'  duration={dur}s")

    else:
        print(f"  Net file not found: {net_file}")

    # Read routes from .rou.xml
    if rou_file and os.path.exists(rou_file):
        print(f"\n  ── Routes in {os.path.basename(rou_file)} ──")
        try:
            rou_tree = ET.parse(rou_file)
            rou_root = rou_tree.getroot()

            vtypes = rou_root.findall("vType")
            print(f"  Vehicle types ({len(vtypes)}):")
            for vt in vtypes:
                print(f"    '{vt.get('id')}'")

            routes = rou_root.findall("route")
            print(f"\n  Routes ({len(routes)}):")
            for r in routes:
                print(f"    '{r.get('id')}': edges = {r.get('edges')}")

            # Count vehicles
            vehicles = rou_root.findall("vehicle")
            print(f"\n  Vehicles scheduled: {len(vehicles)}")
            if vehicles:
                types_found = set(v.get("type", "?") for v in vehicles[:20])
                print(f"  Vehicle types in file: {types_found}")
                # Show emergency vehicles
                em_vehs = [v for v in vehicles
                           if v.get("type", "") == "emergency"]
                print(f"  Emergency vehicles in file: {len(em_vehs)}")
                if em_vehs:
                    print(f"  First emergency: id={em_vehs[0].get('id')} "
                          f"route={em_vehs[0].get('route')} "
                          f"depart={em_vehs[0].get('depart')}")
        except ET.ParseError as e:
            print(f"  Could not parse route file: {e}")
    else:
        print(f"\n  Route file not found: {rou_file}")
        # List all files in SUMO workdir
        print(f"\n  Files in Sumo files folder:")
        for f in sorted(os.listdir(SUMO_WORKDIR)):
            print(f"    {f}")


def read_from_traci():
    """Start SUMO and read everything via TraCI."""
    print("\n" + "="*60)
    print("  Reading via TraCI (live SUMO instance)")
    print("="*60)

    os.chdir(SUMO_WORKDIR)
    cmd = [SUMO_BINARY, "-c", SUMO_CFG,
           "--start", "--quit-on-end",
           "--no-warnings", "--duration-log.disable"]

    try:
        traci.start(cmd)
        traci.simulationStep()

        print("\n  ── Edge IDs (via TraCI) ──")
        edges = [e for e in traci.edge.getIDList()
                 if not e.startswith(":")]
        for e in sorted(edges):
            print(f"    '{e}'")

        print("\n  ── Route IDs ──")
        for r in traci.route.getIDList():
            edges_in_route = traci.route.getEdges(r)
            print(f"    '{r}': {edges_in_route}")

        print("\n  ── Vehicle Type IDs ──")
        for vt in traci.vehicletype.getIDList():
            print(f"    '{vt}'")

        print("\n  ── Traffic Light IDs ──")
        for tl in traci.trafficlight.getIDList():
            phase = traci.trafficlight.getPhase(tl)
            state = traci.trafficlight.getRedYellowGreenState(tl)
            print(f"    '{tl}'  current phase={phase}  state='{state}'")

        print("\n  ── Lanes (first 20) ──")
        lanes = traci.lane.getIDList()
        for lane in sorted(lanes)[:20]:
            if not lane.startswith(":"):
                edge = traci.lane.getEdgeID(lane)
                print(f"    Lane '{lane}'  (edge: '{edge}')")

        traci.close()

    except Exception as e:
        print(f"  TraCI error: {e}")
        print("  Falling back to XML-only reading.")
        try:
            traci.close()
        except Exception:
            pass


def main():
    print("\n  SUMO Network Inspector")
    print("  Reads your actual edge/route names")
    print("  Use these names in train_fixed.py\n")

    # Try XML first (no SUMO needed)
    read_from_xml()

    # Then try TraCI for live data
    try:
        read_from_traci()
    except Exception as e:
        print(f"\n  TraCI reading skipped: {e}")

    print("\n" + "="*60)
    print("  COPY THE EDGE NAMES ABOVE INTO train_fixed2.py")
    print("  Look for edges that connect to your central junction")
    print("="*60)


if __name__ == "__main__":
    main()
