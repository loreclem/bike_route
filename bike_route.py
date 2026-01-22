#!/usr/bin/env python3

import argparse
import osmnx as ox
import networkx as nx
import gpxpy
import gpxpy.gpx
import requests
import time
import sqlite3
from pathlib import Path
from shapely.geometry import LineString, Point
from shapely.ops import substring
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import sys
from math import radians, cos, sin, sqrt, atan2
from matplotlib.colors import Normalize

# Configure logging to standard output
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration & globals
# -----------------------------
ox.settings.log_console = True # logging
# exclude these highways when looking for the shortest path
FORBIDDEN_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "path", "track", "footway", "steps"}
# allow these surfaces when looking for shortest path
ALLOWED_SURFACES = ["asphalt", "paved", "concrete", "gravel", "compacted", "ground"]
# filename of the local sqlite file for storing elevation data
CACHE_DB = Path("elevation_cache.sqlite")
# improves quality of the routing. If the output looks strange, try to set this to False
use_virtual_waypoints = True
# grow map region by `buffer` degrees in order to improve routing capabilities
buffer = 0.05 

# ------------------------------
# Elevation fetching and caching
# ------------------------------

def init_elevation_cache():
    '''
    Helper function to initialise the sqlite
    database to store the local cache of the elevation
    data retrieved from the opentopodata.org API.
    '''
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS elevation (
            lat REAL, lon REAL, elevation REAL,
            PRIMARY KEY (lat, lon)
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Elevation cache initialized at {CACHE_DB}")

def round_coord(lat, lon, ndigits=5):
    return round(lat, ndigits), round(lon, ndigits)

def fetch_elevations_opentopodata_cached(points, dataset="srtm90m", batch_size=100, pause=1.0):
    '''
    It looks up the elevation for a givin set of lat/lon points.
    First, it looks in the local cache file to see if the elevation for
    a given point has already been fatched. If yes, it returns it.
    If not, it fetches it from the opentopodata.org API (be aware of the limitation
    in the amount of allowed queries) and stores it locally into the sqlite database.
    '''
    rounded_points = [round_coord(lat, lon) for lat, lon in points]
    
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    
    elevations = [None] * len(rounded_points)
    to_query = []
    query_indices = []

    for i, (lat, lon) in enumerate(rounded_points):
        cur.execute("SELECT elevation FROM elevation WHERE lat=? AND lon=?", (lat, lon))
        row = cur.fetchone()
        if row:
            elevations[i] = row[0]
        else:
            to_query.append((lat, lon))
            query_indices.append(i)

    logger.info(f"Elevation: {len(rounded_points) - len(to_query)} points from cache, {len(to_query)} points to fetch.")

    if to_query:
        url = f"https://api.opentopodata.org/v1/{dataset}"
        for i in range(0, len(to_query), batch_size):
            batch = to_query[i:i+batch_size]
            logger.info(f"Fetching elevation batch {i//batch_size + 1}/{(len(to_query)-1)//batch_size + 1}...")
            locations = "|".join(f"{lat},{lon}" for lat, lon in batch)
            response = requests.get(url, params={"locations": locations}, timeout=30)
            response.raise_for_status()
            data = response.json()["results"]
            
            for idx_in_batch, r in enumerate(data):
                ele = r["elevation"]
                orig_idx = query_indices[i + idx_in_batch]
                elevations[orig_idx] = ele
                cur.execute("INSERT OR REPLACE INTO elevation VALUES (?, ?, ?)", (batch[idx_in_batch][0], batch[idx_in_batch][1], ele))
            conn.commit()
            if i + batch_size < len(to_query): 
                time.sleep(pause)
            
    conn.close()
    return elevations

# -----------------------------
# Routing helpers
# -----------------------------

def route_to_coords(G, route, spacing=50):
    '''
    It transofrms an OSMNx `route` objet to a list of
    lat/lon coordinates that can then be easily exported as
    a GPX file.
    The spacing of the points is givin in meters.
    '''
    all_coords = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if not edge_data: continue
        data = min(edge_data.values(), key=lambda d: d.get("length", 0))
        
        geom = data.get("geometry", LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]))

        u_xy = (G.nodes[u]['x'], G.nodes[u]['y'])
        if Point(geom.coords[0]).distance(Point(u_xy)) > Point(geom.coords[-1]).distance(Point(u_xy)):
            geom = LineString(list(geom.coords)[::-1])

        length_m = data.get("length", geom.length * 111320)
        num_steps = max(int(length_m / spacing), 1)
        for i in range(num_steps):
            point = geom.interpolate(i / num_steps, normalized=True)
            all_coords.append((point.y, point.x))
            
    all_coords.append((G.nodes[route[-1]]['y'], G.nodes[route[-1]]['x']))
    return all_coords



def add_waypoint_node(G, lat, lon):
    '''
    It creates a virtual OSM node at the exact lat/lon
    position of start, end or waypoint. It improves the precision
    of the final route.
    '''
    u, v, key = ox.distance.nearest_edges(G, lon, lat)
    data = G.get_edge_data(u, v, key)
    geom = data.get("geometry", LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]))
    
    cut_dist = geom.project(Point(lon, lat))
    point_on_line = geom.interpolate(cut_dist)
    
    node_id = max(G.nodes) + random.randint(1000, 9999)
    G.add_node(node_id, x=point_on_line.x, y=point_on_line.y)

    g1, g2 = substring(geom, 0, cut_dist), substring(geom, cut_dist, geom.length)
    total_m = data.get("length", geom.length * 111320)
    l1 = (g1.length / geom.length) * total_m if geom.length > 0 else 0
    l2 = total_m - l1

    for start, end, dist, g in [(u, node_id, l1, g1), (node_id, v, l2, g2)]:
        G.add_edge(start, end, length=dist, geometry=g)
        G.add_edge(end, start, length=dist, geometry=g)
    
    logger.debug(f"Added virtual node {node_id} at ({lat}, {lon})")
    return node_id

def is_allowed_road(edge_data):
    '''
    It checks whether an OSM edge is valid, i.e.
    it is not listed in FORBIDDEN_HIGHWAYS and the surface
    correspond to one of the ALLOWED_SURFACES.
    See global configuration at the beginning of the file.
    '''
    highway = edge_data.get("highway", "")
    if isinstance(highway, list): highway = highway[0]
    if highway in FORBIDDEN_HIGHWAYS: return False
    surface = str(edge_data.get("surface", "")).lower()
    return surface == "nan" or surface == "" or any(s in surface for s in ALLOWED_SURFACES)

# -----------------------------
# Main processing
# -----------------------------

def compute_route(start, end, waypoints, output_gpx):
    '''
    The main processing function taks as input the `start` and `end` points
    in lat/lon. Optionally, waypoints can be given.
    It computes the shortest path between start and end, passing by all the
    waypoints, if any, following the order in which they have been provided.
    Finally, an output GPX file is produced.
    '''
    all_coords = []
    total_length = 0.0
    points = [start] + waypoints + [end]
    
    logger.info(f"Starting route computation for {len(points)} waypoints...")
    
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        logger.info(f"Leg {i+1}/{len(points)-1}: ({lat1}, {lon1}) -> ({lat2}, {lon2})")
        
        n, s, e, w = max(lat1, lat2)+buffer, min(lat1, lat2)-buffer, max(lon1, lon2)+buffer, min(lon1, lon2)-buffer
        
        G = ox.graph_from_bbox((w, s, e, n), network_type="drive", simplify=True)
        initial_edges = len(G.edges)
        
        rem = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not is_allowed_road(d)]
        for edge in rem: G.remove_edge(*edge)
        logger.info(f"  Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges ({initial_edges - len(G.edges)} filtered).")

        src = add_waypoint_node(G, lat1, lon1) if use_virtual_waypoints else ox.distance.nearest_nodes(G, lon1, lat1)
        dst = add_waypoint_node(G, lat2, lon2) if use_virtual_waypoints else ox.distance.nearest_nodes(G, lon2, lat2)
        
        try:
            route = nx.shortest_path(G, src, dst, weight="length")
            leg_dist = nx.path_weight(G, route, weight="length")
            total_length += leg_dist
            logger.info(f"  Shortest path found! Leg distance: {leg_dist/1000:.2f} km")
            
            coords = route_to_coords(G, route)
            all_coords.extend(coords if i == 0 else coords[1:])
        except nx.NetworkXNoPath:
            logger.error(f"  No path found for leg {i+1}! Check forbidden highway settings or buffer size.")

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    for lat, lon in all_coords: seg.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    with open(output_gpx, "w") as f: f.write(gpx.to_xml())
    logger.info(f"GPX file saved to {output_gpx}")
    return total_length

def add_elevation_to_gpx_and_plot(input_gpx_path, total_length, smoothing_window=10, diff_threshold=1):
    '''
    It takes a GPX file as input and, for each lat/lon point found in the file,
    it queries its elevation. This information is then added to a new file, which
    is an identical copy of the input file, but with the added elevation information.
    Furthermore, a PNG plot with the elevation profile and the total distance, and the
    positive and negative elevation difference is produced.
    `smoothing_window` is used to smooth elevation data along the GPX track
    `diff_threshold` is used to improved the precision of the height elevation difference,
    by ignoring differences between two consecutive points which are less than this threshold.
    '''
    logger.info(f"Starting elevation enrichment for {input_gpx_path}...")
    input_path = Path(input_gpx_path)
    output_gpx_path = input_path.with_name(input_path.stem + "_ele.gpx")
    output_png_path = input_path.with_suffix(".png")

    with open(input_gpx_path, "r") as f: gpx = gpxpy.parse(f)
    gpx_out = deepcopy(gpx)

    points, point_refs = [], []
    for track in gpx_out.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
                point_refs.append(point)

    elevations = fetch_elevations_opentopodata_cached(points)
    for pt, ele in zip(point_refs, elevations):
        if ele is not None: pt.elevation = float(ele)

    ele_array = np.array([e if e is not None else 0 for e in elevations])
    ele_smooth = np.convolve(ele_array, np.ones(smoothing_window)/smoothing_window, mode='same') if smoothing_window > 1 else ele_array

    ascent, descent = 0.0, 0.0
    for e1, e2 in zip(ele_smooth[:-1], ele_smooth[1:]):
        diff = e2 - e1
        if abs(diff) >= diff_threshold:
            if diff > 0: ascent += diff
            else: descent -= diff

    with open(output_gpx_path, "w") as f: f.write(gpx_out.to_xml())
    logger.info(f"Enriched GPX saved as {output_gpx_path}")

    # Plotting
    logger.info("Generating elevation profile plot...")
    dist_array = [0.0]
    for i in range(1, len(points)):
        # Haversine-based distance calculation for the x-axis
        dist_array.append(dist_array[-1] + ox.distance.euclidean(points[i-1][0], points[i-1][1], points[i][0], points[i][1]) * 111320 / 1000)

    plt.figure(figsize=(10, 4))
    crop_plot = round(smoothing_window / 2.0)
    sc = plt.scatter(dist_array[crop_plot:crop_plot*-1], ele_smooth[crop_plot:crop_plot*-1], c=ele_smooth[crop_plot:crop_plot*-1], cmap="terrain", norm=Normalize(0, 2000), s=10)
    plt.plot(dist_array[crop_plot:crop_plot*-1], ele_smooth[crop_plot:crop_plot*-1], color="black", linewidth=0.5, alpha=0.6)
    plt.colorbar(sc, label="Elevation (m)")
    plt.text(0.02, 0.95, f"Dist: {total_length/1000:.1f}km\nAsc: +{ascent:.0f}m\nDesc: -{descent:.0f}m", 
             transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel("Distance (km)"); plt.ylabel("Elevation (m)"); plt.title(input_path.name)
    plt.savefig(output_png_path, dpi=150); plt.close()
    logger.info(f"Elevation profile plot saved as {output_png_path}")
    print(f"\n--- Summary ---\nTotal Distance: {total_length/1000:.2f} km\nTotal Ascent: +{ascent:.0f} m\nTotal Descent: -{descent:.0f} m\n---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-lat", type=float, required=True)
    parser.add_argument("--start-lon", type=float, required=True)
    parser.add_argument("--end-lat", type=float, required=True)
    parser.add_argument("--end-lon", type=float, required=True)
    parser.add_argument("--waypoints", type=float, nargs="*", default=[])
    parser.add_argument("--output", default="route.gpx")
    parser.add_argument("--ele", action="store_true")
    args = parser.parse_args()

    init_elevation_cache()
    wpts = [(args.waypoints[i], args.waypoints[i+1]) for i in range(0, len(args.waypoints), 2)]
    
    total_len = compute_route((args.start_lat, args.start_lon), (args.end_lat, args.end_lon), wpts, args.output)
    if args.ele:
        add_elevation_to_gpx_and_plot(args.output, total_len)

# Sample usage:

# $ python bike_route.py --start-lat 46.097796 --start-lon 8.931909 --end-lat 45.992768 --end-lon 9.238409 --waypoints 46.030575  8.920863  46.024374  9.074518 --output my_route.gpx --ele
