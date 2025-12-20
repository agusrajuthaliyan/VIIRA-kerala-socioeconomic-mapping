# 05_process_opencellid.py
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

# --- INPUT / OUTPUT PATHS (project-relative) ---
F1 = "data/raw/opencellid/404.csv"
F2 = "data/raw/opencellid/405.csv"

STATES_GEO = "data/raw/opencellid/states_india.geojson"  # optional
KERALA_GEO = "data/raw/kerala.geojson"
OUT_DIR = "data/processed"

COMBINED_CSV = os.path.join(OUT_DIR, "opencellid_india_combined.csv")
KERALA_CSV = os.path.join(OUT_DIR, "opencellid_kerala.csv")
KERALA_GEOJSON = os.path.join(OUT_DIR, "opencellid_kerala.geojson")
KERALA_2023 = os.path.join(OUT_DIR, "opencellid_kerala_2023.geojson")

os.makedirs(OUT_DIR, exist_ok=True)

# --- helper to detect lat/lon column names ---
def detect_lat_lon(cols):
    low = [c.lower() for c in cols]
    lat_candidates = [c for c in cols if c.lower() in ("lat","latitude","lat_deg","latd","latitude_deg")]
    lon_candidates = [c for c in cols if c.lower() in ("lon","longitude","lon_deg","lng","long","longitude_deg")]
    return (lat_candidates[0] if lat_candidates else None,
            lon_candidates[0] if lon_candidates else None)

# --- Step 1: combine 404 + 405 into one CSV (chunked to avoid OOM) ---
def combine_csvs(f1, f2, out_path, chunksize=1_000_000):
    print("Combining CSVs:", f1, "and", f2)
    first = True
    for fn in (f1, f2):
        if not os.path.exists(fn):
            print("Warning: file not found:", fn)
            continue
        for chunk in pd.read_csv(fn, chunksize=chunksize, low_memory=True):
            if first:
                chunk.to_csv(out_path, index=False, mode="w")
                first = False
            else:
                chunk.to_csv(out_path, index=False, mode="a", header=False)
    if not os.path.exists(out_path):
        raise FileNotFoundError("Combined CSV not created. Check input files.")
    print("Saved combined CSV:", out_path)

# --- Step 2: quick inspect combined header and timestamps ---
def inspect_combined(path, nrows=5):
    print("Inspecting combined CSV header and sample rows...")
    dfh = pd.read_csv(path, nrows=nrows)
    print("Columns:", dfh.columns.tolist())
    lat_col, lon_col = detect_lat_lon(dfh.columns.tolist())
    print("Detected lat/lon columns:", lat_col, lon_col)
    # check for time cols
    time_cols = [c for c in dfh.columns if c.lower() in ("created","updated","lastseen","timestamp","time")]
    print("Possible time columns:", time_cols)
    return lat_col, lon_col, time_cols

# --- Step 3: filter, parse timestamps, convert to GeoDataFrame, clip to Kerala ---
def filter_and_clip(combined_csv, lat_col, lon_col, time_cols, kerala_geojson, out_csv, out_geojson, out_geojson_2023=None):
    print("Loading combined CSV (may take a while)...")
    df = pd.read_csv(combined_csv, low_memory=False)
    print("Rows before cleaning:", len(df))

    # Ensure lat/lon exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Latitude/Longitude columns not found. Detected: {lat_col}, {lon_col}. Inspect CSV header.")

    # convert to numeric & drop invalid coordinates
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df[df[lat_col].notnull() & df[lon_col].notnull()].copy()
    print("Rows after dropping invalid coords:", len(df))

    # parse timestamps if present
    parsed = False
    for col in time_cols:
        if col in df.columns:
            try:
                df[col + "_dt"] = pd.to_datetime(pd.to_numeric(df[col], errors='coerce'), unit='s', errors='coerce')
                print(f"Parsed {col} to datetime. Min: {df[col + '_dt'].min()}, Max: {df[col + '_dt'].max()}")
                parsed = True
            except Exception as e:
                print("Could not parse time column", col, ":", e)

    # create GeoDataFrame (WGS84)
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])], crs="EPSG:4326")
    print("Converted to GeoDataFrame. CRS:", gdf.crs, "Rows:", len(gdf))

    # Load Kerala geometry and ensure same CRS
    kerala = gpd.read_file(kerala_geojson).to_crs(gdf.crs)
    # spatial clip using bounding box filter first for speed
    minx, miny, maxx, maxy = kerala.total_bounds
    gdf = gdf[(gdf.geometry.x >= (minx-0.5)) & (gdf.geometry.x <= (maxx+0.5)) & (gdf.geometry.y >= (miny-0.5)) & (gdf.geometry.y <= (maxy+0.5))].copy()
    print("After bbox pre-filter rows:", len(gdf))

    # exact clip
    gdf_ker = gdf[gdf.within(kerala.unary_union)].copy()
    print("After precise within() clip rows (Kerala):", len(gdf_ker))

    # save outputs
    print("Saving Kerala CSV and GeoJSON...")
    gdf_ker.drop(columns="geometry").to_csv(out_csv, index=False)
    gdf_ker.to_file(out_geojson, driver="GeoJSON")
    print("Saved:", out_csv, out_geojson)

    # create 2023 subset if Updated_dt / Created_dt present
    if parsed:
        # prefer Updated_dt if exists, else Created_dt
        time_dt_cols = [c for c in gdf_ker.columns if c.endswith("_dt")]
        if time_dt_cols:
            # choose a column which seems most populated
            colcounts = {c: gdf_ker[c].notnull().sum() for c in time_dt_cols}
            best_col = max(colcounts, key=colcounts.get)
            print("Using", best_col, "to filter year=2023 (counts:", colcounts[best_col], ")")
            subset_2023 = gdf_ker[gdf_ker[best_col].dt.year == 2023].copy()
            print("Rows observed in 2023:", len(subset_2023))
            if out_geojson_2023:
                subset_2023.to_file(out_geojson_2023, driver="GeoJSON")
                print("Saved 2023 subset:", out_geojson_2023)
        else:
            print("No parsed datetime columns available to filter for 2023.")
    else:
        print("No timestamp parsed; skipping 2023 subset.")

    return gdf_ker

# --- Main flow ---
def main():
    # Combine CSVs (if combined not exists)
    if not os.path.exists(COMBINED_CSV):
        combine_csvs(F1, F2, COMBINED_CSV)
    else:
        print("Combined CSV already exists:", COMBINED_CSV)

    # Inspect and detect lat/lon
    lat_col, lon_col, time_cols = inspect_combined(COMBINED_CSV)

    # If detect failed on small sample, inspect full header
    if lat_col is None or lon_col is None:
        full_head = pd.read_csv(COMBINED_CSV, nrows=2)
        lat_col, lon_col = detect_lat_lon(full_head.columns.tolist())
        print("Fallback detected lat/lon:", lat_col, lon_col)
        if lat_col is None or lon_col is None:
            raise SystemExit("Could not detect latitude/longitude columns. Please inspect CSV header.")

    # run filter & clip
    gdf_ker = filter_and_clip(COMBINED_CSV, lat_col, lon_col, time_cols, KERALA_GEO, KERALA_CSV, KERALA_GEOJSON, KERALA_2023)

    print("Done. Kerala towers saved. Example rows:")
    print(gdf_ker.head().to_string())

if __name__ == "__main__":
    main()
