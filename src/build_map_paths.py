"""
One-off authoring tool for the Grain Belt Atlas dashboard's map -- NOT part
of the tested pipeline (extract_*.py -> build_dataset.py -> train.py ->
simulate.py) and not run by CI. Needs `pip install shapely` (not in
requirements.txt -- only this script uses it).

Fetches a public-domain world countries GeoJSON, extracts our 9 target
countries + regional context countries, simplifies, projects (simple
equirectangular with latitude-cosine x-compression), and writes
map_paths.json: ready-to-embed SVG <path> d-strings plus centroid
coordinates. This gets baked into the dashboard's HTML at authoring time so
the published page needs zero runtime network/tile requests -- required by
the Artifact CSP, which does not allow loading map tiles or GeoJSON from an
external host at runtime.

Country outlines are simplified for display only, not survey-accurate.
"""
import json
import requests
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union

URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
data = requests.get(URL, timeout=30).json()

TARGETS = {
    "Afghanistan": "AFG", "Pakistan": "PAK", "Iran": "IRN", "Kazakhstan": "KAZ",
    "Uzbekistan": "UZB", "Tajikistan": "TJK", "Turkey": "TUR", "Ukraine": "UKR",
    "Netherlands": "NLD",
    # Expansion set -- now spans every populated continent, so the map is a
    # world map, not a regional one (see BBOX removal below).
    "China": "CHN", "India": "IND", "United States of America": "USA",
    "Canada": "CAN", "Australia": "AUS", "Argentina": "ARG", "Egypt": "EGY",
    "Morocco": "MAR", "Kyrgyzstan": "KGZ", "Azerbaijan": "AZE", "Georgia": "GEO",
    "France": "FRA", "Germany": "DEU", "Turkmenistan": "TKM",
}

# Regional context: everything else, styled neutral, for orientation.
by_name = {f["properties"].get("name"): f for f in data["features"]}

# Projection: equirectangular, x = lon * cos(ref_lat), y = -lat, then scale/translate.
import math
REF_LAT = 40.0
COS_REF = math.cos(math.radians(REF_LAT))

def project(lon, lat):
    return (lon * COS_REF, -lat)

def simplify_geom(geom, tol):
    g = shape(geom)
    return g.simplify(tol, preserve_topology=True)

def geom_to_path(geom, tol):
    g = simplify_geom(geom, tol)
    polys = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    d_parts = []
    for poly in polys:
        if poly.is_empty:
            continue
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            coords = list(ring.coords)
            if len(coords) < 3:
                continue
            pts = [project(lon, lat) for lon, lat in coords]
            d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
            d_parts.append(d)
    return " ".join(d_parts)

target_paths = {}
target_centroids = {}
for name, iso in TARGETS.items():
    feat = by_name[name]
    geom = feat["geometry"]
    target_paths[iso] = geom_to_path(geom, tol=0.03)
    c = shape(geom).centroid
    target_centroids[iso] = project(c.x, c.y)

# Context countries: keep separate (no union -- avoids invalid-geometry issues
# from the full-world dataset). Targets now span every populated continent, so
# this is a world map -- only Antarctica is excluded (by southern latitude,
# not name, since coordinate-cutting is what actually controls file size/
# viewBox, and some sub-Antarctic islands share a feature with mainland
# countries). Coarser tolerance since these are just orientation fill.
MIN_LAT = -58
context_paths = []
for name, feat in by_name.items():
    if name in TARGETS:
        continue
    g = shape(feat["geometry"])
    b = g.bounds  # (minx,miny,maxx,maxy)
    if b[3] < MIN_LAT:  # entirely south of MIN_LAT -- Antarctica itself
        continue
    try:
        p = geom_to_path(feat["geometry"], tol=0.08)
        if p:
            context_paths.append(p)
    except Exception as e:
        print("skip", name, e)
context_path = " ".join(context_paths)

out = {
    "target_paths": target_paths,
    "target_centroids": {k: [round(v[0],2), round(v[1],2)] for k,v in target_centroids.items()},
    "context_path": context_path,
    "ref_lat_cos": COS_REF,
}
with open("data/processed/map_paths.json", "w") as f:
    json.dump(out, f)

for iso, p in target_paths.items():
    print(iso, len(p), "chars")
print("context path:", len(context_path), "chars")
