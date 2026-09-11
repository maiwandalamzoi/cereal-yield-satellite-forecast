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

# Context countries: keep separate (no union -- avoids invalid-geometry issues from
# the full-world dataset), restricted to a bounding box around our region so we're
# not carrying Antarctica etc, coarser tolerance since these are just orientation fill.
BBOX = (-15, 10, 95, 65)  # lon_min, lat_min, lon_max, lat_max
context_paths = []
for name, feat in by_name.items():
    if name in TARGETS:
        continue
    g = shape(feat["geometry"])
    b = g.bounds  # (minx,miny,maxx,maxy)
    if b[2] < BBOX[0] or b[0] > BBOX[2] or b[3] < BBOX[1] or b[1] > BBOX[3]:
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
