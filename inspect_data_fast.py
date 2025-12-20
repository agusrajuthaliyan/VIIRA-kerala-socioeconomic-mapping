import geopandas as gpd
import pandas as pd

path = "data/processed/merged_features_simplified.geojson"
try:
    # Read only first row to get columns/schema
    gdf = gpd.read_file(path, rows=1)
    print("Columns found:")
    for col in gdf.columns:
        print(f" - {col}")
except Exception as e:
    print(f"Error: {e}")
