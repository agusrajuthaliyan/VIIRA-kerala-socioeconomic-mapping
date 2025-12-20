import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import os

# ---------- CONFIG ----------
TOWER_CSV_PATH = "data/processed/opencellid_kerala.csv" 
GRID_PATH = "data/processed/merged_features.geojson"
OUTPUT_PATH = "data/processed/merged_features_dist.geojson"

def main():
    print("🚀 Starting Distance Calculation (using existing Kerala CSV)...")

    # 1. Load Towers
    if not os.path.exists(TOWER_CSV_PATH):
        print(f"❌ Error: Could not find {TOWER_CSV_PATH}")
        return

    print(f"   - Reading {TOWER_CSV_PATH}...")
    df = pd.read_csv(TOWER_CSV_PATH)
    
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # FIX: Rename 'long' to 'lon' if it exists
    if 'long' in df.columns:
        df = df.rename(columns={'long': 'lon'})
    
    # Verify columns exist now
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("❌ Error: CSV must have 'lat' and 'lon' (or 'long') columns.")
        print(f"   Found columns: {list(df.columns)}")
        return

    print(f"   - Loaded {len(df):,} towers.")

    # 2. Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    towers_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 3. Load Grid
    print(f"   - Loading Grid: {GRID_PATH}")
    if not os.path.exists(GRID_PATH):
        print("❌ Grid file not found!")
        return
    grid = gpd.read_file(GRID_PATH)

    # 4. Project to Metric (UTM 43N)
    print("   - Projecting to UTM for accurate calculation...")
    towers_m = towers_gdf.to_crs(epsg=32643)
    grid_m = grid.to_crs(epsg=32643)

    # 5. Calculate Distance
    print("   - Calculating distances (using spatial index)...")
    tower_multipoint = towers_m.geometry.unary_union

    def get_dist(point):
        nearest = nearest_points(point, tower_multipoint)[1]
        return point.distance(nearest) / 1000.0 # Convert meters to km

    grid_m["dist_to_tower_km"] = grid_m.centroid.apply(get_dist)

    # 6. Save
    final_grid = grid_m.to_crs(epsg=4326)
    final_grid.to_file(OUTPUT_PATH, driver="GeoJSON")
    
    print("✅ SUCCESS!")
    print(f"   - New feature 'dist_to_tower_km' saved to: {OUTPUT_PATH}")
    print("   - You can now run the app.")

if __name__ == "__main__":
    main()