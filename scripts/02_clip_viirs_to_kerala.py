import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# ---- PATHS (use the exact project-relative paths) ----
VIIRS_PATH = "data/raw/viirs/viirs_2023_median_masked.tif"
KERALA_GEOJSON = "data/raw/kerala.geojson"
OUT_DIR = "data/processed"
OUT_RASTER = os.path.join(OUT_DIR, "viirs_kerala_2023.tif")

os.makedirs(OUT_DIR, exist_ok=True)

def clip_viirs():
    print("🔹 Opening VIIRS raster:", VIIRS_PATH)
    with rasterio.open(VIIRS_PATH) as src:
        print("  CRS:", src.crs)
        print("  Width x Height:", src.width, "x", src.height)
        print("  dtype:", src.dtypes)
        nodata = src.nodata
        print("  nodata:", nodata)

        print("🔹 Loading Kerala polygon:", KERALA_GEOJSON)
        kerala = gpd.read_file(KERALA_GEOJSON)

        # ensure same CRS
        if kerala.crs != src.crs:
            print("  Reprojecting Kerala polygon to raster CRS...")
            kerala = kerala.to_crs(src.crs)

        geoms = kerala.geometry.values
        print("🔹 Clipping... (this may take a minute or two)")

        out_img, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "count": out_img.shape[0]
        })

    print("🔹 Writing clipped raster to:", OUT_RASTER)
    with rasterio.open(OUT_RASTER, "w", **out_meta) as dst:
        dst.write(out_img)

    print("✅ Clipped VIIRS saved:", OUT_RASTER)

if __name__ == "__main__":
    clip_viirs()
