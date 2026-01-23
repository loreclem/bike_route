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
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
from math import radians, cos, sin, sqrt, atan2
from matplotlib.colors import Normalize

# Configure logging to standard output
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our helpers
from .utils import (
    init_elevation_cache, 
    fetch_elevations_opentopodata_cached, 
    is_allowed_road,
    add_waypoint_node,
    CACHE_DB,
    buffer,
    use_virtual_waypoints,
    route_to_coords
)

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

def main():
    # Sample usage:
    # $ python bike_route.py --start-lat 46.097796 --start-lon 8.931909 --end-lat 45.992768 --end-lon 9.238409 --waypoints 46.030575  8.920863  46.024374  9.074518 --output my_route.gpx --ele
    
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

if __name__ == "__main__":
    main()
