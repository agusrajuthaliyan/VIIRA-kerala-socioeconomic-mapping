# 04_compute_zonal_stats.py
import os
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

GRID_PATH = "data/processed/kerala_grid_1km.geojson"
RASTER_PATH = "data/processed/viirs_kerala_2023.tif"
OUT_PATH = "data/processed/kerala_grid_viirs_features.geojson"

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print("Loading grid:", GRID_PATH)
    grid = gpd.read_file(GRID_PATH)
    print("Grid rows:", len(grid), "  CRS:", grid.crs)

    print("Opening raster to inspect:", RASTER_PATH)
    with rasterio.open(RASTER_PATH) as src:
        raster_crs = src.crs
        nodata = src.nodata
        print("Raster CRS:", raster_crs, " nodata:", nodata)

    # reproject grid to raster CRS if needed
    if grid.crs != raster_crs:
        print("Reprojecting grid to raster CRS...")
        grid = grid.to_crs(raster_crs)

    print("Computing zonal stats (mean, sum, std, count). This may take a few minutes...")
    stats = zonal_stats(
        grid["geometry"],
        RASTER_PATH,
        stats=["mean", "sum", "std", "count"],
        nodata=nodata,
        all_touched=False,
        geojson_out=False,
        raster_out=False,
    )

    print("Converting stats to DataFrame...")
    stats_df = pd.DataFrame(stats).fillna(0)  # fill NaNs with 0 (safe for many use-cases)
    # ensure same length
    if len(stats_df) != len(grid):
        raise RuntimeError("Length mismatch between stats and grid: {} vs {}".format(len(stats_df), len(grid)))

    # attach columns to grid
    grid = grid.reset_index(drop=True)
    grid["nl_mean"] = stats_df["mean"].astype(float)
    grid["nl_sum"]  = stats_df["sum"].astype(float)
    grid["nl_std"]  = stats_df["std"].astype(float)
    grid["nl_count"] = stats_df["count"].astype(int)

    # Some basic derived columns
    # avoid divide by zero
    grid["nl_mean_per_pixel"] = np.where(grid["nl_count"]>0, grid["nl_sum"]/grid["nl_count"], 0)

    print("Saving grid with VIIRS features to:", OUT_PATH)
    grid.to_file(OUT_PATH, driver="GeoJSON")
    print("✅ Saved:", OUT_PATH)
    print("Sample rows:")
    print(grid[["nl_mean","nl_sum","nl_std","nl_count"]].head())

if __name__ == "__main__":
    main()
