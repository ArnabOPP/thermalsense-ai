import requests
import json
from pathlib import Path

# Get Kolkata wards from Overpass API
query = """
[out:json][timeout:60];
area[name='Kolkata'][admin_level=4]->.kolkata;
(
  relation[admin_level=9](area.kolkata);
  relation[admin_level=10](area.kolkata);
);
out geom;
"""

print("Fetching ward boundaries...")
try:
    r = requests.post('https://overpass-api.de/api/interpreter', data=query, timeout=90)
    data = r.json()
    n = len(data["elements"])
    print(f"Found {n} ward relations")

    Path('frontend/public').mkdir(exist_ok=True)
    with open('frontend/public/kolkata_wards_raw.json', 'w') as f:
        json.dump(data, f)
    print("Saved to frontend/public/kolkata_wards_raw.json")
except Exception as e:
    print(f"Failed: {e}")
    print("Will use grid approach instead")