"""
Clip feature matrix pixels to actual city boundary polygon.
Saves a filtered version with only pixels inside the boundary.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, shape

ROOT = Path('D:/ISROPS1/data-pipeline')

cities = {
    'kolkata': {
        'parquet': ROOT / 'outputs/exports/kolkata/feature_matrix_kolkata_ALL.parquet',
        'boundary': ROOT / 'frontend/public/kolkata_boundary.geojson',
        'out': ROOT / 'outputs/exports/kolkata/feature_matrix_kolkata_clipped.parquet',
    }
}

for city, cfg in cities.items():
    print(f"\nClipping {city}...")
    
    # Load boundary
    with open(cfg['boundary']) as f:
        geojson = json.load(f)
    boundary = shape(geojson)
    print(f"  Boundary type: {boundary.geom_type}")
    print(f"  Boundary bounds: {boundary.bounds}")
    
    # Load pixels
    df = pd.read_parquet(cfg['parquet'])
    df = df[df['lst_celsius'] > 10].copy()
    print(f"  Total pixels: {len(df):,}")
    
    # Clip to boundary
    print("  Clipping pixels to boundary (this takes a moment)...")
    mask = df.apply(lambda row: boundary.contains(Point(row['centroid_lon'], row['centroid_lat'])), axis=1)
    df_clipped = df[mask].copy()
    print(f"  Pixels inside boundary: {len(df_clipped):,}")
    
    # Save
    df_clipped.to_parquet(cfg['out'], index=False)
    print(f"  Saved: {cfg['out']}")