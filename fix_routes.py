"""
fix_routes.py
══════════════════════════════════════════════════════════════════════
Fixes the emergency routes in all route files.

Your actual edges are:
  Incoming:  north_to_center, south_to_center, east_to_center, west_to_center
  Outgoing:  center_to_south, center_to_north, center_to_west, center_to_east

The generated files used wrong outgoing edge names (south_exit, north_exit etc.)
This script fixes all route files in your Sumo files folder.

Run:
    python fix_routes.py
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"

# ── Correct routes based on your actual network edges ─────────────────
CORRECT_ROUTES = {
    "north_south": "north_to_center center_to_south",
    "south_north": "south_to_center center_to_north",
    "east_west":   "east_to_center center_to_west",
    "west_east":   "west_to_center center_to_east",
    # Emergency routes — same paths, just labelled separately
    "emergency_ns": "north_to_center center_to_south",
    "emergency_sn": "south_to_center center_to_north",
    "emergency_ew": "east_to_center center_to_west",
    "emergency_we": "west_to_center center_to_east",
}

# Route files to fix
ROUTE_FILES = [
    "routes_india_normal.rou.xml",
    "routes_india_offpeak.rou.xml",
    "routes_india_peak.rou.xml",
    "routes_real_normal.rou.xml",
    "routes_real_offpeak.rou.xml",
    "routes_real_peak.rou.xml",
    "routes.rou.xml",
]


def fix_route_file(filepath):
    if not os.path.exists(filepath):
        print(f"  Skipping (not found): {os.path.basename(filepath)}")
        return False

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Parse error in {os.path.basename(filepath)}: {e}")
        return False

    changed = False

    for route_elem in root.findall("route"):
        rid   = route_elem.get("id", "")
        edges = route_elem.get("edges", "")

        if rid in CORRECT_ROUTES:
            correct = CORRECT_ROUTES[rid]
            if edges.strip() != correct:
                print(f"    Fixed route '{rid}':")
                print(f"      was:  '{edges.strip()}'")
                print(f"      now:  '{correct}'")
                route_elem.set("edges", correct)
                changed = True
        else:
            # Check if edges contain any wrong names
            wrong_names = ["south_exit", "north_exit", "east_exit", "west_exit",
                           "south_entry", "north_entry", "east_entry", "west_entry"]
            if any(w in edges for w in wrong_names):
                print(f"    Warning: route '{rid}' contains unknown edge: '{edges}'")

    # Add missing emergency routes if they don't exist
    existing_route_ids = {r.get("id") for r in root.findall("route")}
    for rid, edges in CORRECT_ROUTES.items():
        if rid not in existing_route_ids:
            new_route = ET.SubElement(root, "route")
            new_route.set("id", rid)
            new_route.set("edges", edges)
            print(f"    Added missing route: '{rid}'")
            changed = True

    # Make sure emergency vehicle type exists
    existing_vtypes = {v.get("id") for v in root.findall("vType")}
    if "emergency" not in existing_vtypes:
        em_type = ET.Element("vType")
        em_type.set("id", "emergency")
        em_type.set("accel", "3.5")
        em_type.set("decel", "5.0")
        em_type.set("sigma", "0.2")
        em_type.set("length", "5.5")
        em_type.set("minGap", "1.5")
        em_type.set("maxSpeed", "22.22")
        em_type.set("guiShape", "emergency")
        em_type.set("color", "1,0,0")
        em_type.set("priority", "10")
        # Insert at beginning (before routes and vehicles)
        root.insert(0, em_type)
        print(f"    Added missing vType: 'emergency'")
        changed = True

    if changed:
        # Write back
        xml_str = minidom.parseString(
            ET.tostring(root, encoding='unicode')
        ).toprettyxml(indent="    ")
        lines = [l for l in xml_str.split('\n') if l.strip()]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  ✓ Fixed: {os.path.basename(filepath)}")
        return True
    else:
        print(f"  OK (no changes needed): {os.path.basename(filepath)}")
        return False


def verify_routes(filepath):
    """Quick verification that the fixed routes are valid."""
    if not os.path.exists(filepath):
        return

    valid_edges = {
        "north_to_center", "south_to_center",
        "east_to_center",  "west_to_center",
        "center_to_north", "center_to_south",
        "center_to_east",  "center_to_west",
    }

    tree = ET.parse(filepath)
    root = tree.getroot()
    all_ok = True

    for route in root.findall("route"):
        rid   = route.get("id", "")
        edges = route.get("edges", "").split()
        for edge in edges:
            if edge not in valid_edges:
                print(f"  ✗ Route '{rid}' still has invalid edge: '{edge}'")
                all_ok = False

    if all_ok:
        routes_found = [r.get("id") for r in root.findall("route")]
        print(f"  ✓ All routes valid: {routes_found}")


def main():
    print("\n" + "="*60)
    print("  Fixing SUMO route files")
    print("  Your network edges:")
    print("    Incoming: north_to_center, south_to_center,")
    print("              east_to_center,  west_to_center")
    print("    Outgoing: center_to_north, center_to_south,")
    print("              center_to_east,  center_to_west")
    print("="*60)

    fixed_count = 0
    for filename in ROUTE_FILES:
        filepath = os.path.join(SUMO_WORKDIR, filename)
        print(f"\n  Processing: {filename}")
        if fix_route_file(filepath):
            fixed_count += 1

    print(f"\n  Fixed {fixed_count} file(s)")

    # Verify the main route file
    main_route = os.path.join(SUMO_WORKDIR, "routes_india_normal.rou.xml")
    if os.path.exists(main_route):
        print(f"\n  Verifying routes_india_normal.rou.xml:")
        verify_routes(main_route)

    print("\n" + "="*60)
    print("  Done. Now run: python train_fixed.py")
    print("="*60)


if __name__ == "__main__":
    main()
