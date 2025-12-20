# scripts/simplify_geojson.py
import geopandas as gpd
import os

IN = r"data/processed/merged_features.geojson"
OUT = r"data/processed/merged_features_simplified.geojson"

print("Loading (this may take a moment)...")
g = gpd.read_file(IN)

# pick a reasonable UTM / metric CRS for Kerala; you can auto-detect but this EPSG works
g_utm = g.to_crs(epsg=32643)    # if wrong, change later to your zone

print("Simplifying geometries...")
g_utm["geometry"] = g_utm.geometry.simplify(tolerance=50, preserve_topology=True)

print("Converting back to WGS84 and saving:", OUT)
g_simpl = g_utm.to_crs(epsg=4326)
g_simpl.to_file(OUT, driver="GeoJSON")
print("Done. Simplified file ready:", OUT)
print("Original size:", os.path.getsize(IN), "bytes")
print("Simplified size:", os.path.getsize(OUT), "bytes")
