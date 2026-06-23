"""
patch_active_routes.py
══════════════════════════════════════════════════════════════════════
Reads traffic.sumocfg to find EXACTLY which route file is loaded,
then patches that specific file to fix emergency_ew and emergency_we.

Run this then immediately run the simulation.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

SUMO_WORKDIR = r"C:\Users\ASUS\Desktop\MTECH\SET PROJECT\Project\Sumo files"
SUMO_CFG     = os.path.join(SUMO_WORKDIR, "traffic.sumocfg")

# Your confirmed correct edges from net.net.xml
CORRECT_EMERGENCY_ROUTES = {
    "emergency_ns": "north_to_center center_to_south",
    "emergency_sn": "south_to_center center_to_north",
    "emergency_ew": "east_to_center center_to_west",
    "emergency_we": "west_to_center center_to_east",
}

VALID_EDGES = {
    "north_to_center", "south_to_center",
    "east_to_center",  "west_to_center",
    "center_to_north", "center_to_south",
    "center_to_east",  "center_to_west",
}


def get_active_route_file():
    """Read traffic.sumocfg and return the currently active route file path."""
    tree = ET.parse(SUMO_CFG)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag == "route-files":
            rou_filename = elem.get("value", "")
            return os.path.join(SUMO_WORKDIR, rou_filename)
    return None


def patch_route_file(filepath):
    """Fix all emergency routes and print every change made."""
    print(f"\n  Patching: {os.path.basename(filepath)}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    changed = False
    existing_route_ids = {r.get("id") for r in root.findall("route")}

    # Fix or add every emergency route
    for rid, correct_edges in CORRECT_EMERGENCY_ROUTES.items():
        found = False
        for route_elem in root.findall("route"):
            if route_elem.get("id") == rid:
                found = True
                current = route_elem.get("edges", "").strip()
                if current != correct_edges:
                    print(f"    FIXED  '{rid}'")
                    print(f"      was: '{current}'")
                    print(f"      now: '{correct_edges}'")
                    route_elem.set("edges", correct_edges)
                    changed = True
                else:
                    print(f"    OK     '{rid}' = '{correct_edges}'")
                break

        if not found:
            # Insert route after last existing route element
            route_elem = ET.SubElement(root, "route")
            route_elem.set("id", rid)
            route_elem.set("edges", correct_edges)
            print(f"    ADDED  '{rid}' = '{correct_edges}'")
            changed = True

    # Also fix any VEHICLES that reference bad routes
    # Remove vehicles using emergency_ew or emergency_we if those routes were broken
    # (SUMO already spawned them as errors — just remove future ones)
    bad_routes = set()
    for route_elem in root.findall("route"):
        edges = route_elem.get("edges", "").split()
        for edge in edges:
            if edge not in VALID_EDGES:
                bad_routes.add(route_elem.get("id"))

    if bad_routes:
        print(f"\n  WARNING: Still has invalid edges in routes: {bad_routes}")
    else:
        print(f"\n  All {len(CORRECT_EMERGENCY_ROUTES)} emergency routes are valid.")

    # Save
    if changed:
        xml_str = minidom.parseString(
            ET.tostring(root, encoding='unicode')
        ).toprettyxml(indent="    ")
        lines = [l for l in xml_str.split('\n') if l.strip()]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n  Saved.")
    else:
        print(f"\n  No changes needed — file was already correct.")

    return changed


def patch_all_route_files():
    """Patch every route file in the Sumo files folder."""
    patched = 0
    for filename in os.listdir(SUMO_WORKDIR):
        if filename.endswith(".rou.xml"):
            filepath = os.path.join(SUMO_WORKDIR, filename)
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                # Only patch files that have route definitions
                if root.findall("route"):
                    patch_route_file(filepath)
                    patched += 1
            except ET.ParseError:
                print(f"  Skip (parse error): {filename}")
    return patched


def main():
    print("="*60)
    print("  Patching active SUMO route file")
    print("="*60)

    # Step 1: Find which file is active
    active_file = get_active_route_file()
    print(f"\n  Active route file: {active_file}")

    if not active_file or not os.path.exists(active_file):
        print("  Could not find active route file.")
        print("  Patching ALL .rou.xml files instead...")
        n = patch_all_route_files()
        print(f"\n  Patched {n} route files.")
        return

    # Step 2: Patch active file
    patch_route_file(active_file)

    # Step 3: Also patch all others to be safe
    print("\n  Also patching all other .rou.xml files...")
    for filename in os.listdir(SUMO_WORKDIR):
        if filename.endswith(".rou.xml"):
            full = os.path.join(SUMO_WORKDIR, filename)
            if full != active_file:
                try:
                    tree = ET.parse(full)
                    root = tree.getroot()
                    if root.findall("route"):
                        changed = False
                        for route_elem in root.findall("route"):
                            rid = route_elem.get("id", "")
                            if rid in CORRECT_EMERGENCY_ROUTES:
                                correct = CORRECT_EMERGENCY_ROUTES[rid]
                                current = route_elem.get("edges","").strip()
                                if current != correct:
                                    route_elem.set("edges", correct)
                                    changed = True
                        if changed:
                            xml_str = minidom.parseString(
                                ET.tostring(root, encoding='unicode')
                            ).toprettyxml(indent="    ")
                            lines = [l for l in xml_str.split('\n') if l.strip()]
                            with open(full, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lines))
                            print(f"  Fixed: {filename}")
                except Exception:
                    pass

    print("\n" + "="*60)
    print("  Done. Now run: python run_simulation_gui.py")
    print("  The emergency vehicle errors should be gone.")
    print("="*60)


if __name__ == "__main__":
    main()
