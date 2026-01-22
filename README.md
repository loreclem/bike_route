# bike_route 🚴‍♂️

`bike_route` is a command-line tool to generate **bike-friendly routes** between two points (optionally via waypoints), export them as **GPX files**, and enrich them with **elevation data and profiles**.

It is designed for cyclists who want:
- routes that follow real roads from OpenStreetMap,
- avoidance of unsuitable road types (e.g. motorways),
- GPX output compatible with GPS devices and navigation apps,
- realistic elevation gain/loss and a clear elevation profile.

The tool relies on **OSMnx**, **NetworkX**, and the **OpenTopoData** free elevation API (with local caching). OSMnx, in turn, relies on the overpass-turbo.de API for fetching OSM data.


## Features

- 🚴 Bike-oriented routing using OpenStreetMap data  
- 📍 Support for intermediate waypoints  
- 🗺️ GPX export following actual road geometry  
- ⛰️ Elevation enrichment using OpenTopoData (SRTM 90 m)  
- 📈 Automatic elevation profile plot (PNG)  
- 💾 Local SQLite cache for elevation queries  
- ⚙️ Simple and scriptable CLI interface  


## Command-line interface (CLI)

    usage: bike_route.py [options]

Required arguments:

    --start-lat FLOAT Latitude of start point
    --start-lon FLOAT Longitude of start point
    --end-lat FLOAT Latitude of end point
    --end-lon FLOAT Longitude of end point

Optional arguments:

    --waypoints FLOAT ... Intermediate waypoints as lat lon pairs
    --output FILE.gpx Output GPX file (default: route.gpx)
    --ele Add elevation to GPX and generate PNG elevation profile


Waypoints must be provided as **pairs of latitude and longitude**.

### Routing strategy

In order to reduce the size of the OSMnx graph to be downloaded and to speed-up the processing time, the algorithm will compute the shortest path between start and waypoint1, then from waypoint1 to the next and so on, until the end is reached. For long routes (e.g. > 50 km), it is advisable to provide some waypoints. This will increase the number of requests to overpass-turbo.de API via OSMnx, but the overall performance improves.

## Example usage


    $ python bike_route.py \
      --start-lat 42.702442 --start-lon 9.452907 \
      --end-lat 42.268426 --end-lon 8.693644 \
      --waypoints 42.564580 8.976084 42.518302 8.693800 \
      --output bike_route_corsica.gpx \
      --ele

This will:

- Compute a bike-friendly route via the given waypoints
- Write a GPX file following real roads
- Add elevation data to the GPX
- Generate an elevation profile PNG
- Print distance and total ascent/descent to stdout


## Example output

[GPX file without elevation](https://raw.githubusercontent.com/loreclem/bike_route/refs/heads/main/sample_output/my_route.gpx)
[GPX file wit elevation](https://raw.githubusercontent.com/loreclem/bike_route/refs/heads/main/sample_output/my_route_ele.gpx)

![Elevation profile](https://github.com/loreclem/bike_route/blob/main/sample_output/my_route.png?raw=true)

## Requirements

- Python ≥ 3.9
- Linux (tested)
- Internet connection (for OpenStreetMap & OpenTopoData)

## Main Python dependencies:

    osmnx
    networkx
    gpxpy
    shapely
    pyproj
    requests
    matplotlib
    numpy

## Installation

Clone the repository:
    
    git clone https://github.com/loreclem/bike_route.git
    cd bike_route
    
Install dependencies (example using pip):
    
    pip install -r requirements.txt

## Notes

Elevation data is cached locally in elevation_cache.sqlite
OpenTopoData usage is rate-limited; caching avoids repeated queries
Routing quality depends on OpenStreetMap tagging completeness
Some of the code has been written or proofed with the help of AI
This is an early release — feedback and contributions are welcome
	

## License

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
Please refer to the repository for license details.

## Acknowledgements

- OpenStreetMap contributors
- OSMnx project and the underlying overpass-turbo.de API
- OpenTopoData API
