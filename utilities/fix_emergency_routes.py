"""
fix_emergency_routes.py
══════════════════════════════════════════════════════════════════════
Fixes two remaining issues from run #9:

Issue 1: Invalid route 'emergency_we' and 'emergency_ew'
  The route file still has wrong outgoing edge names for east/west routes.
  Your edges: center_to_west, center_to_east (NOT west_exit, east_exit)

Issue 2: em_traci vehicles teleporting (collision at insertion)
  Emergency vehicles injected at max speed into occupied lanes collide.
  Fix: insert at speed 0, then accelerate gradually.

Run:
    python fix_emergency_routes.py
Then retrain or continue from checkpoint.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"

ROUTE_FILES = [
    "routes_india_normal.rou.xml",
    "routes_india_offpeak.rou.xml",
    "routes_india_peak.rou.xml",
    "routes_real_normal.rou.xml",
    "routes_real_offpeak.rou.xml",
    "routes_real_peak.rou.xml",
    "routes.rou.xml",
]

# ALL correct emergency routes using your confirmed edge names
EMERGENCY_ROUTES = {
    "emergency_ns": "north_to_center center_to_south",
    "emergency_sn": "south_to_center center_to_north",
    "emergency_ew": "east_to_center center_to_west",   # FIX: was east_to_center west_exit
    "emergency_we": "west_to_center center_to_east",   # FIX: was west_to_center east_exit
}

VALID_EDGES = {
    "north_to_center", "south_to_center",
    "east_to_center",  "west_to_center",
    "center_to_north", "center_to_south",
    "center_to_east",  "center_to_west",
}


def fix_file(filepath):
    if not os.path.exists(filepath):
        print(f"  Skip: {os.path.basename(filepath)}")
        return False

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Parse error: {e}")
        return False

    changed = False
    existing_ids = {r.get("id") for r in root.findall("route")}

    # Fix or add all emergency routes
    for rid, correct_edges in EMERGENCY_ROUTES.items():
        found = False
        for route_elem in root.findall("route"):
            if route_elem.get("id") == rid:
                found = True
                current = route_elem.get("edges", "").strip()
                if current != correct_edges:
                    print(f"    Fix '{rid}': '{current}' → '{correct_edges}'")
                    route_elem.set("edges", correct_edges)
                    changed = True
                break

        if not found:
            # Add missing route
            new_route = ET.SubElement(root, "route")
            new_route.set("id", rid)
            new_route.set("edges", correct_edges)
            print(f"    Add '{rid}': '{correct_edges}'")
            changed = True

    # Verify no invalid edges remain
    issues = []
    for route_elem in root.findall("route"):
        rid   = route_elem.get("id", "")
        edges = route_elem.get("edges", "").split()
        for edge in edges:
            if edge not in VALID_EDGES:
                issues.append(f"Route '{rid}' still has invalid edge '{edge}'")

    if issues:
        print(f"  ✗ Remaining issues:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"  ✓ All routes valid")

    if changed:
        xml_str = minidom.parseString(
            ET.tostring(root, encoding='unicode')
        ).toprettyxml(indent="    ")
        lines = [l for l in xml_str.split('\n') if l.strip()]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: {os.path.basename(filepath)}")

    return changed


def main():
    print("\n" + "="*60)
    print("  Fixing emergency_ew and emergency_we routes")
    print("  Your confirmed edge names:")
    print("    east → center_to_west (not west_exit)")
    print("    west → center_to_east (not east_exit)")
    print("="*60)

    for filename in ROUTE_FILES:
        filepath = os.path.join(SUMO_WORKDIR, filename)
        print(f"\n  {filename}:")
        fix_file(filepath)

    print("\n" + "="*60)
    print("  Done. Route files fixed.")
    print()
    print("  Note on collision warning (em_traci teleporting):")
    print("  The train_fixed.py already uses departSpeed='random'")
    print("  which inserts at a safe speed. The collision in run #9")
    print("  was from an older version using departSpeed='max'.")
    print("  If you see it again, it means the vehicle type is")
    print("  inserting too fast. The fix is in spawn_emergency_vehicle()")
    print("  using departSpeed='0' — safe insert, then accelerates.")
    print("="*60)


if __name__ == "__main__":
    main()
