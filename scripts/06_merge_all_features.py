# 06_merge_all_features.py  -- adapted to your OpenCellID columns
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

GRID_PATH = "data/processed/kerala_grid_viirs_features.geojson"
# TOWERS_PATH = "data/processed/opencellid_kerala.geojson"
TOWERS_PATH = "data/processed/opencellid_kerala_2023.geojson"

OUT_PATH = "data/processed/merged_features.geojson"

def to_utm(gdf):
    centroid = gdf.geometry.unary_union.centroid
    lon = centroid.x
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone
    return gdf.to_crs(epsg)

def safe_load():
    grid = gpd.read_file(GRID_PATH)
    towers = gpd.read_file(TOWERS_PATH)
    print("Loaded grid rows:", len(grid), "towers rows:", len(towers))
    return grid, towers

def spatial_join_and_aggregate(grid_utm, towers_utm):
    # spatial join: towers -> grid index
    joined = gpd.sjoin(towers_utm, grid_utm, how="left", predicate="intersects")
    print("Joined rows (tower->grid):", len(joined))

    # tower counts per grid cell
    tower_counts = joined.groupby("index_right").size().rename("tower_count")
    feats = pd.DataFrame(tower_counts)

    # numeric aggregates (range, avgsignal, sample)
    for col in ("range", "avgsignal", "sample"):
        if col in joined.columns:
            feats[f"mean_{col}"] = joined.groupby("index_right")[col].mean()

    # radio/type counts (one-hot like)
    if "radio" in joined.columns:
        radio_counts = (joined
                        .groupby(["index_right", joined["radio"].astype(str).str.upper()])
                        .size()
                        .unstack(fill_value=0))
        radio_counts.columns = [f"radio_{str(c).lower()}" for c in radio_counts.columns]
        feats = feats.join(radio_counts, how="left")

    # join back to grid
    grid_utm = grid_utm.reset_index(drop=True)
    grid_utm = grid_utm.join(feats, how="left")
    # fill NaNs
    grid_utm["tower_count"] = grid_utm["tower_count"].fillna(0).astype(int)
    for c in feats.columns:
        if c not in grid_utm.columns:
            continue
        if grid_utm[c].dtype.kind in "fiu":
            grid_utm[c] = grid_utm[c].fillna(0)

    return grid_utm, joined

def compute_nearest(grid_utm, towers_utm):
    coords = np.array([(p.x, p.y) for p in towers_utm.geometry])
    if len(coords) == 0:
        grid_utm["nearest_tower_m"] = np.nan
        return grid_utm
    tree = cKDTree(coords)
    centroids = np.array([(g.centroid.x, g.centroid.y) for g in grid_utm.geometry])
    dists, _ = tree.query(centroids, k=1)
    grid_utm["nearest_tower_m"] = dists
    return grid_utm

def main():
    print("Loading data...")
    grid, towers = safe_load()

    # Ensure towers have expected columns 'long','lat' -> geometry already exists in your geojson
    # Project both to UTM for meter units
    print("Projecting to UTM...")
    grid_utm = to_utm(grid)
    towers_utm = towers.to_crs(grid_utm.crs)

    # Aggregate towers into grid
    grid_utm, joined = spatial_join_and_aggregate(grid_utm, towers_utm)

    # Add area and density
    grid_utm["area_km2"] = grid_utm.geometry.area / 1e6
    grid_utm["tower_density"] = grid_utm["tower_count"] / grid_utm["area_km2"].replace({0: np.nan})

    # nearest tower distance
    grid_utm = compute_nearest(grid_utm, towers_utm)

    # add simple combined features
    # nl_mean already exists from VIIRS zonal stats; ensure it exists
    if "nl_mean" not in grid_utm.columns:
        grid_utm["nl_mean"] = 0.0

    # nl per tower (avoid div by zero)
    grid_utm["nl_per_tower"] = grid_utm["nl_sum"] / (grid_utm["tower_count"] + 1)

    # Save result
    print("Saving merged features to:", OUT_PATH)
    grid_utm.to_file(OUT_PATH, driver="GeoJSON")
    print("Done. Saved merged features.")

if __name__ == "__main__":
    main()
