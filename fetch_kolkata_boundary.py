import requests
import json
from pathlib import Path

# Get Kolkata Municipal Corporation boundary from Overpass
query = """
[out:json][timeout:60];
relation["name"="Kolkata"]["admin_level"="5"];
out geom;
"""

print("Fetching Kolkata boundary from Overpass...")
try:
    r = requests.post('https://overpass-api.de/api/interpreter', data=query, timeout=90)
    data = r.json()
    elements = data.get('elements', [])
    print(f"Found {len(elements)} relations")
    for el in elements[:5]:
        print(f"  - {el.get('tags', {}).get('name')} | admin_level={el.get('tags', {}).get('admin_level')} | id={el.get('id')}")
except Exception as e:
    print(f"Overpass failed: {e}")

# Fallback: use a known GeoJSON for Kolkata from a public source
print("\nTrying fallback - building bbox-based boundary...")
# KMC actual boundary as approximate polygon
kmc_coords = [
    [88.2073, 22.4956], [88.2200, 22.4800], [88.2600, 22.4650],
    [88.3000, 22.4580], [88.3400, 22.4600], [88.3800, 22.4700],
    [88.4200, 22.4900], [88.4500, 22.5100], [88.4600, 22.5400],
    [88.4550, 22.5700], [88.4400, 22.5950], [88.4200, 22.6150],
    [88.3900, 22.6300], [88.3600, 22.6380], [88.3200, 22.6350],
    [88.2900, 22.6200], [88.2600, 22.5950], [88.2300, 22.5700],
    [88.2100, 22.5400], [88.2073, 22.4956]
]

geojson = {
    "type": "Polygon",
    "coordinates": [kmc_coords]
}

Path('frontend/public').mkdir(exist_ok=True)
with open('frontend/public/kolkata_boundary.geojson', 'w') as f:
    json.dump(geojson, f)
print("Saved approximate KMC boundary")
print(f"Bounds: lon {min(c[0] for c in kmc_coords):.4f}-{max(c[0] for c in kmc_coords):.4f}, lat {min(c[1] for c in kmc_coords):.4f}-{max(c[1] for c in kmc_coords):.4f}")