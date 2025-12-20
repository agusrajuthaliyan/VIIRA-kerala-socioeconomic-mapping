# 03_create_1km_grid.py
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from pyproj import CRS

# Paths (project-relative)
KERALA_GEOJSON = "data/raw/kerala.geojson"
OUT_GRID = "data/processed/kerala_grid_1km.geojson"

def choose_utm_epsg(gdf):
    # choose UTM zone by the geometry centroid longitude
    centroid = gdf.geometry.unary_union.centroid
    lon = centroid.x
    lat = centroid.y
    # pyproj CRS helper: get UTM for lon/lat
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={int((lon + 180) / 6) + 1} +datum=WGS84 +units=m +no_defs")
    # Convert to EPSG code if possible
    try:
        epsg = utm_crs.to_epsg()
    except Exception:
        epsg = None
    return epsg if epsg is not None else utm_crs.to_string()

def create_grid(cell_size_m=1000):
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    print("Loading Kerala polygon:", KERALA_GEOJSON)
    kerala = gpd.read_file(KERALA_GEOJSON)

    # ensure it's in geographic coords lon/lat (EPSG:4326) before choosing UTM
    if kerala.crs is None:
        kerala.set_crs(epsg=4326, inplace=True)
    elif kerala.crs.to_string().lower().find("4326") == -1 and not kerala.crs.is_geographic:
        kerala = kerala.to_crs(epsg=4326)

    # pick UTM zone EPSG code
    utm_epsg = choose_utm_epsg(kerala)
    print("Selected projection (UTM):", utm_epsg)

    # reproject to UTM for meter-based grid
    kerala_utm = kerala.to_crs(epsg=utm_epsg) if isinstance(utm_epsg, int) else kerala.to_crs(utm_epsg)

    minx, miny, maxx, maxy = kerala_utm.total_bounds
    print("Kerala bounds (meters):", minx, miny, maxx, maxy)

    # optionally pad the bounding box slightly to ensure full coverage
    pad = 0
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    # create grid cells
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    print(f"Creating grid: {len(xs)} x {len(ys)} = approx {len(xs)*len(ys)} cells (before clipping)")

    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + cell_size_m, y + cell_size_m))

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=kerala_utm.crs)

    # intersect with Kerala polygon to clip grid to boundary
    grid_clipped = gpd.overlay(grid, kerala_utm, how="intersection")

    print("Grid cells after clipping:", len(grid_clipped))

    # compute centroids and area for possible use
    grid_clipped["centroid_x"] = grid_clipped.geometry.centroid.x
    grid_clipped["centroid_y"] = grid_clipped.geometry.centroid.y
    grid_clipped["area_m2"] = grid_clipped.geometry.area

    # save in UTM projection (recommended) but also save as EPSG:4326 optionally
    grid_clipped.to_file(OUT_GRID, driver="GeoJSON")
    print("Saved grid →", OUT_GRID)
    print("Grid CRS:", grid_clipped.crs)
    print("Done.")

if __name__ == "__main__":
    create_grid(cell_size_m=1000)
