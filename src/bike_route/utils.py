import sqlite3
import requests
import time
import logging
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, Point
import osmnx as ox
import random
from shapely.ops import substring


logger = logging.getLogger(__name__)

# -----------------------------
# This file contains the routing
# and helpers functions, as well
# as the main configuration parameters
# -----------------------------

# -------------
# Configuration
# -------------
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

# -----------------
# Helpers functions
# -----------------

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
