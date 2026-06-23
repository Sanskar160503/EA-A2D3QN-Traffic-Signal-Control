"""
osm_intersection.py
══════════════════════════════════════════════════════════════════════
Extract a real Indian intersection from OpenStreetMap and convert
it to a SUMO network file.

This replaces your synthetic 4-way intersection with a real
intersection geometry from Chennai or Bangalore.

No login required. Uses OSM Overpass API (free).

Usage:
    python osm_intersection.py --city chennai    # Anna Salai intersection
    python osm_intersection.py --city bangalore  # MG Road intersection
    python osm_intersection.py --lat 13.0827 --lon 80.2707 --name my_intersection

Requirements:
    pip install requests
    SUMO must be installed (uses netconvert)
"""

import os
import sys
import json
import subprocess
import argparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ── Predefined Indian intersections ───────────────────────────────────
# These are real intersections chosen because they have clear 4-way
# geometry suitable for single-intersection traffic signal control

INDIAN_INTERSECTIONS = {
    "chennai": {
        "name":        "Anna Salai - Nandanam Junction, Chennai",
        "lat":         13.0358,
        "lon":         80.2510,
        "radius":      150,     # meters to extract
        "description": "Major 4-way signalized intersection on Anna Salai",
        "city":        "Chennai, Tamil Nadu",
        "source":      "OpenStreetMap contributors",
    },
    "chennai_kk": {
        "name":        "Kathipara Junction, Chennai",
        "lat":         13.0067,
        "lon":         80.2206,
        "radius":      200,
        "description": "Complex cloverleaf, use for multi-agent extension",
        "city":        "Chennai, Tamil Nadu",
        "source":      "OpenStreetMap contributors",
    },
    "bangalore": {
        "name":        "MG Road - Brigade Road Junction, Bangalore",
        "lat":         12.9747,
        "lon":         77.6101,
        "radius":      150,
        "description": "Central Bangalore signalized intersection",
        "city":        "Bangalore, Karnataka",
        "source":      "OpenStreetMap contributors",
    },
    "bangalore_silk": {
        "name":        "Silk Board Junction, Bangalore",
        "lat":         12.9172,
        "lon":         77.6233,
        "radius":      200,
        "description": "One of Bangalore's most congested intersections",
        "city":        "Bangalore, Karnataka",
        "source":      "OpenStreetMap contributors",
    },
    "mumbai": {
        "name":        "Dadar Junction, Mumbai",
        "lat":         19.0178,
        "lon":         72.8478,
        "radius":      150,
        "description": "Major Mumbai arterial intersection",
        "city":        "Mumbai, Maharashtra",
        "source":      "OpenStreetMap contributors",
    },
}


def download_osm_data(lat, lon, radius=150, output_file="intersection.osm"):
    """
    Download OSM data for a bounding box around the intersection.
    Uses Overpass API — completely free, no registration.
    """
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    # Compute bounding box
    # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111*cos(lat) km
    import math
    dlat = radius / 111000.0
    dlon = radius / (111000.0 * math.cos(math.radians(lat)))

    south = lat - dlat
    north = lat + dlat
    west  = lon - dlon
    east  = lon + dlon

    bbox  = f"{south},{west},{north},{east}"

    # Overpass query — gets roads and signals
    query = f"""
    [out:xml][timeout:30];
    (
        way["highway"]({bbox});
        node["highway"="traffic_signals"]({bbox});
        relation["type"="restriction"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """

    overpass_url = "https://overpass-api.de/api/interpreter"

    print(f"[OSM] Downloading intersection data...")
    print(f"  Center: ({lat}, {lon})")
    print(f"  Radius: {radius}m")
    print(f"  Bbox:   {bbox}")

    try:
        resp = requests.post(overpass_url, data={"data": query}, timeout=60)
        resp.raise_for_status()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(resp.text)

        size_kb = os.path.getsize(output_file) / 1024
        print(f"[OSM] Downloaded: {output_file} ({size_kb:.1f} KB)")
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"[OSM] Download failed: {e}")
        print("  Try again in a few seconds (Overpass rate limit)")
        return None


def convert_osm_to_sumo(osm_file, output_dir=".", net_name="real_intersection"):
    """
    Convert OSM file to SUMO network using netconvert.
    netconvert is included with SUMO installation.
    """
    net_file  = os.path.join(output_dir, f"{net_name}.net.xml")
    poly_file = os.path.join(output_dir, f"{net_name}.poly.xml")

    # Find netconvert
    netconvert_paths = [
        r"C:\Program Files\Eclipse\Sumo\bin\netconvert.exe",
        r"C:\Program Files (x86)\Eclipse\Sumo\bin\netconvert.exe",
        "netconvert",   # if in PATH
    ]
    netconvert = None
    for p in netconvert_paths:
        if os.path.exists(p) or p == "netconvert":
            netconvert = p
            break

    if not netconvert:
        print("[netconvert] Not found. Install SUMO and try again.")
        print("  Or manually run:")
        print(f"  netconvert --osm-files {osm_file} "
              f"--output-file {net_file} "
              f"--geometry.remove --roundabouts.guess "
              f"--ramps.guess --junctions.join "
              f"--tls.guess-signals --tls.discard-simple "
              f"--tls.join --no-internal-links")
        return None

    cmd = [
        netconvert,
        "--osm-files",          osm_file,
        "--output-file",        net_file,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--no-internal-links",
        "--keep-edges.by-vclass", "passenger,emergency,bus,motorcycle",
        "--type-files",
        r"C:\Program Files\Eclipse\Sumo\data\typemap\osmNetconvert.typ.xml",
    ]

    print(f"\n[netconvert] Converting OSM to SUMO network...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            size_kb = os.path.getsize(net_file) / 1024 if os.path.exists(net_file) else 0
            print(f"[netconvert] Success: {net_file} ({size_kb:.1f} KB)")
            return net_file
        else:
            print(f"[netconvert] Error: {result.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        print("[netconvert] Timeout — try with smaller radius")
        return None
    except FileNotFoundError:
        print(f"[netconvert] Not found at: {netconvert}")
        return None


def generate_sumocfg(net_file, route_file, output_dir=".",
                     name="real_intersection"):
    """
    Generate a .sumocfg file for the real intersection.
    Drop-in replacement for your current traffic.sumocfg.
    """
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(route_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <max-depart-delay value="600"/>
    </processing>
    <output>
        <summary-output value="{name}_summary.xml"/>
        <tripinfo-output value="{name}_tripinfo.xml"/>
    </output>
    <gui_only>
        <start value="true"/>
    </gui_only>
</configuration>"""

    cfg_file = os.path.join(output_dir, f"{name}.sumocfg")
    with open(cfg_file, 'w') as f:
        f.write(cfg_content)
    print(f"[SUMO] Config file: {cfg_file}")
    return cfg_file


def print_intersection_info(key):
    """Print info about a predefined intersection."""
    info = INDIAN_INTERSECTIONS.get(key)
    if not info:
        print(f"Unknown intersection: {key}")
        return
    print(f"\n  Intersection: {info['name']}")
    print(f"  City:         {info['city']}")
    print(f"  Coordinates:  ({info['lat']}, {info['lon']})")
    print(f"  Description:  {info['description']}")
    print(f"  Data source:  {info['source']}")
    print(f"\n  Google Maps:  "
          f"https://maps.google.com/?q={info['lat']},{info['lon']}")
    print(f"  OSM:          "
          f"https://www.openstreetmap.org/#map=18/{info['lat']}/{info['lon']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract real Indian intersection from OpenStreetMap for SUMO"
    )
    parser.add_argument("--city",   default="chennai",
                        choices=list(INDIAN_INTERSECTIONS.keys()),
                        help="Predefined Indian intersection")
    parser.add_argument("--lat",    type=float, help="Custom latitude")
    parser.add_argument("--lon",    type=float, help="Custom longitude")
    parser.add_argument("--radius", type=int,   default=150,
                        help="Extraction radius in meters")
    parser.add_argument("--outdir", default=".",
                        help="Output directory")
    parser.add_argument("--info",   action="store_true",
                        help="Print info about intersections only")
    args = parser.parse_args()

    if args.info:
        print("\nAvailable Indian intersections:")
        for key, info in INDIAN_INTERSECTIONS.items():
            print(f"\n  {key}:")
            print_intersection_info(key)
        return

    # Get coordinates
    if args.lat and args.lon:
        lat, lon   = args.lat, args.lon
        name       = f"custom_{lat}_{lon}"
        radius     = args.radius
    else:
        info   = INDIAN_INTERSECTIONS[args.city]
        lat    = info["lat"]
        lon    = info["lon"]
        radius = info["radius"]
        name   = args.city
        print_intersection_info(args.city)

    os.makedirs(args.outdir, exist_ok=True)

    # Download OSM
    osm_file = os.path.join(args.outdir, f"{name}.osm")
    osm_file = download_osm_data(lat, lon, radius, osm_file)
    if not osm_file:
        print("\nFailed to download OSM data.")
        print("Using your existing synthetic network instead.")
        return

    # Convert to SUMO network
    net_file = convert_osm_to_sumo(osm_file, args.outdir, name)
    if not net_file:
        print("\nConversion failed.")
        print("Your existing net.xml will still work.")
        return

    print(f"\n[Done] Real Indian intersection ready:")
    print(f"  Network file: {net_file}")
    print(f"\nTo use in your project:")
    print(f"  1. Copy {name}.net.xml to your Sumo files folder")
    print(f"  2. Update traffic.sumocfg to point to it:")
    print(f'     <net-file value="{name}.net.xml"/>')
    print(f"\nCitation for your paper:")
    if args.city in INDIAN_INTERSECTIONS:
        info = INDIAN_INTERSECTIONS[args.city]
        print(f"  \"{info['name']}\" intersection geometry")
        print(f"  extracted from OpenStreetMap (© OpenStreetMap contributors)")
        print(f"  using SUMO netconvert. Location: {info['city']}.")


if __name__ == "__main__":
    main()
