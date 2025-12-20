import geopandas as gpd
import pandas as pd
import numpy as np

path = "data/processed/merged_features_simplified.geojson"
try:
    gdf = gpd.read_file(path)
    print("Columns:", gdf.columns.tolist())
    
    numeric_cols = gdf.select_dtypes(include=np.number).columns.tolist()
    print("\nNumeric Columns:", numeric_cols)
    
    for col in numeric_cols:
        print(f"\nStats for {col}:")
        print(gdf[col].describe())
        print(f"Zeros: {(gdf[col] == 0).sum()} / {len(gdf)}")
        print(f"NAs: {gdf[col].isna().sum()}")
        # Check unique values for potential discrete/tower columns
        unique_vals = gdf[col].dropna().unique()
        if len(unique_vals) < 50:
            print(f"Unique values (discrete?): {sorted(unique_vals)}")
        else:
            print(f"Feature seems continuous (Unique count: {len(unique_vals)})")

except Exception as e:
    print(f"Error reading file: {e}")
