"""
fix_vtypes.py
══════════════════════════════════════════════════════════════════════
Adds missing vehicle types to all route files.

Error: The vehicle type 'cycle' for vehicle 'cycle_south_north_1359'
       is not known.

The indian_dataset.py added cycle vehicles to the route file but the
vType definition for 'cycle' was missing. This script adds all missing
vTypes to every route file.

Run:
    python fix_vtypes.py
Then:
    python train_fixed.py
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

# All vehicle types your project uses
# These match SUMO's built-in guiShapes
ALL_VTYPES = {
    "car": {
        "accel": "2.6", "decel": "4.5", "sigma": "0.5",
        "length": "4.5", "minGap": "1.5", "maxSpeed": "13.89",
        "guiShape": "passenger", "color": "0.7,0.7,0.7",
    },
    "two_wheeler": {
        "accel": "3.0", "decel": "5.0", "sigma": "0.8",
        "length": "2.2", "minGap": "1.0", "maxSpeed": "11.11",
        "guiShape": "motorcycle", "color": "0.8,0.4,0.0",
    },
    "autorickshaw": {
        "accel": "2.0", "decel": "4.0", "sigma": "0.6",
        "length": "3.5", "minGap": "1.2", "maxSpeed": "8.33",
        "guiShape": "taxi", "color": "1.0,0.8,0.0",
    },
    "bus": {
        "accel": "1.5", "decel": "3.5", "sigma": "0.3",
        "length": "12.0", "minGap": "2.5", "maxSpeed": "11.11",
        "guiShape": "bus", "color": "0.2,0.5,0.2",
    },
    "truck": {
        "accel": "1.2", "decel": "3.0", "sigma": "0.3",
        "length": "10.0", "minGap": "3.0", "maxSpeed": "8.33",
        "guiShape": "truck", "color": "0.5,0.3,0.1",
    },
    "cycle": {
        "accel": "1.2", "decel": "3.0", "sigma": "0.9",
        "length": "1.8", "minGap": "0.8", "maxSpeed": "5.56",
        "guiShape": "bicycle", "color": "0.0,0.6,0.8",
    },
    "emergency": {
        "accel": "3.5", "decel": "5.0", "sigma": "0.2",
        "length": "5.5", "minGap": "1.5", "maxSpeed": "22.22",
        "guiShape": "emergency", "color": "1,0,0", "priority": "10",
    },
}

# Correct emergency routes using your actual edge names
CORRECT_ROUTES = {
    "north_south":   "north_to_center center_to_south",
    "south_north":   "south_to_center center_to_north",
    "east_west":     "east_to_center center_to_west",
    "west_east":     "west_to_center center_to_east",
    "emergency_ns":  "north_to_center center_to_south",
    "emergency_sn":  "south_to_center center_to_north",
    "emergency_ew":  "east_to_center center_to_west",
    "emergency_we":  "west_to_center center_to_east",
}

VALID_EDGES = {
    "north_to_center", "south_to_center",
    "east_to_center",  "west_to_center",
    "center_to_north", "center_to_south",
    "center_to_east",  "center_to_west",
}


def fix_file(filepath):
    if not os.path.exists(filepath):
        print(f"  Skip (not found): {os.path.basename(filepath)}")
        return

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Parse error: {e}")
        return

    changed = False

    # ── Fix vTypes ────────────────────────────────────────────────────
    existing_vtypes = {v.get("id") for v in root.findall("vType")}

    # Find which vehicle types are actually used in this file
    used_types = set()
    for veh in root.findall("vehicle"):
        used_types.add(veh.get("type", ""))
    for veh in root.findall("flow"):
        used_types.add(veh.get("type", ""))

    # Always include emergency type
    used_types.add("emergency")

    # Add missing vTypes that are used
    insert_pos = 0   # insert at start of root
    for vtype_id in sorted(used_types):
        if vtype_id and vtype_id not in existing_vtypes and vtype_id in ALL_VTYPES:
            elem = ET.Element("vType")
            elem.set("id", vtype_id)
            for k, v in ALL_VTYPES[vtype_id].items():
                elem.set(k, v)
            root.insert(insert_pos, elem)
            insert_pos += 1
            existing_vtypes.add(vtype_id)
            print(f"    + Added vType: '{vtype_id}'")
            changed = True

    # Also update existing vTypes that have wrong params (optional)
    for vtype_elem in root.findall("vType"):
        vid = vtype_elem.get("id", "")
        if vid in ALL_VTYPES and not vtype_elem.get("guiShape"):
            for k, v in ALL_VTYPES[vid].items():
                vtype_elem.set(k, v)
            changed = True

    # ── Fix routes ────────────────────────────────────────────────────
    existing_route_ids = {r.get("id") for r in root.findall("route")}

    # Fix existing routes with wrong edges
    for route_elem in root.findall("route"):
        rid   = route_elem.get("id", "")
        edges = route_elem.get("edges", "").strip()

        if rid in CORRECT_ROUTES:
            correct = CORRECT_ROUTES[rid]
            if edges != correct:
                print(f"    ~ Fixed route '{rid}': '{edges}' → '{correct}'")
                route_elem.set("edges", correct)
                changed = True
        else:
            # Check for any invalid edges
            edge_list = edges.split()
            bad = [e for e in edge_list if e not in VALID_EDGES]
            if bad:
                print(f"    ! Route '{rid}' has unknown edges: {bad}")

    # Add missing routes
    for rid, edges in CORRECT_ROUTES.items():
        if rid not in existing_route_ids:
            route_elem = ET.Element("route")
            route_elem.set("id", rid)
            route_elem.set("edges", edges)
            # Insert after vTypes, before vehicles
            # Find position after last vType
            last_vtype_idx = 0
            for i, child in enumerate(root):
                if child.tag == "vType":
                    last_vtype_idx = i
            root.insert(last_vtype_idx + 1, route_elem)
            print(f"    + Added route: '{rid}' → '{edges}'")
            changed = True
            existing_route_ids.add(rid)

    # ── Write back ────────────────────────────────────────────────────
    if changed:
        xml_str = minidom.parseString(
            ET.tostring(root, encoding='unicode')
        ).toprettyxml(indent="    ")
        lines = [l for l in xml_str.split('\n') if l.strip()]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  ✓ Saved: {os.path.basename(filepath)}")
    else:
        print(f"  ✓ No changes needed: {os.path.basename(filepath)}")


def verify(filepath):
    """Quick sanity check after fixing."""
    if not os.path.exists(filepath):
        return
    tree  = ET.parse(filepath)
    root  = tree.getroot()
    vtypes = {v.get("id") for v in root.findall("vType")}
    routes = {r.get("id"): r.get("edges") for r in root.findall("route")}
    vehs   = root.findall("vehicle")

    # Check all vehicle types exist
    issues = []
    for veh in vehs:
        vt = veh.get("type", "")
        if vt and vt not in vtypes:
            issues.append(f"Vehicle '{veh.get('id')}' uses unknown type '{vt}'")

    # Check all route edges are valid
    for rid, edges in routes.items():
        for edge in (edges or "").split():
            if edge not in VALID_EDGES:
                issues.append(f"Route '{rid}' has unknown edge '{edge}'")

    if issues:
        print(f"\n  ✗ Still has issues:")
        for issue in issues[:5]:
            print(f"    {issue}")
    else:
        veh_count  = len(vehs)
        em_vehs    = [v for v in vehs if v.get("type") == "emergency"]
        print(f"\n  ✓ Verified: {veh_count} vehicles, "
              f"{len(em_vehs)} emergency, "
              f"{len(vtypes)} vTypes, "
              f"{len(routes)} routes — all valid")


def main():
    print("\n" + "="*60)
    print("  Fixing vehicle types and routes in all SUMO route files")
    print("="*60)

    for filename in ROUTE_FILES:
        filepath = os.path.join(SUMO_WORKDIR, filename)
        print(f"\n  {filename}:")
        fix_file(filepath)

    # Verify the main file
    main_file = os.path.join(SUMO_WORKDIR, "routes_india_normal.rou.xml")
    print(f"\n  Verification — routes_india_normal.rou.xml:")
    verify(main_file)

    print("\n" + "="*60)
    print("  All done. Now run: python train_fixed.py")
    print("="*60)


if __name__ == "__main__":
    main()
