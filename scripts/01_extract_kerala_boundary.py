import geopandas as gpd
import os

# Paths
GADM_PATH = "data/raw/gadm/gadm41_IND_1.shp"
OUTPUT_PATH = "data/raw/kerala.geojson"

def extract_kerala():
    print("🔹 Loading GADM Level-1 shapefile...")
    gdf = gpd.read_file(GADM_PATH)

    print("🔹 Filtering for Kerala state...")
    kerala = gdf[gdf["NAME_1"].str.lower() == "kerala"]

    if kerala.empty:
        raise ValueError("❌ Kerala not found in GADM Level-1 file. Check 'NAME_1' column.")

    print("🔹 Saving Kerala boundary to:", OUTPUT_PATH)
    kerala.to_file(OUTPUT_PATH, driver="GeoJSON")

    print("✅ Kerala boundary successfully extracted!")

if __name__ == "__main__":
    extract_kerala()
