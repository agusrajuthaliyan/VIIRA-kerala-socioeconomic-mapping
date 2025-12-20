"""
07_model_training.py

- Loads merged_features.geojson
- If a ground-truth raster or CSV is provided, runs supervised regression with spatial CV:
    * Baselines: VIIRS-only vs VIIRS+Towers (ablation)
    * Models: LightGBM, RandomForest
    * Metrics: RMSE, MAE, Pearson r
    * SHAP for LightGBM fused model
- If no ground-truth is provided, runs unsupervised:
    * KMeans (k=3) + HDBSCAN cluster visualizations
    * Compares nl-only clustering vs fused clustering qualitatively
- Saves outputs in results/ and data/models/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ML libs
import lightgbm as lgb

# Paths (change if needed)
MERGED_GEO = "data/processed/merged_features.geojson"
RESULTS_DIR = "results"
MODELS_DIR = "data/models"

# Provide ground truth here (optionally). If raster, set GROUND_TRUTH_RASTER to the file path.
# If CSV (grid_id, target) set GROUND_TRUTH_CSV to path. If both None -> unsupervised.
GROUND_TRUTH_RASTER = None   # e.g., "data/raw/worldpop_poverty_2023.tif"
GROUND_TRUTH_CSV = None      # e.g., "data/raw/poverty_districts_to_grid.csv"

# Modeling settings
N_SPLITS = 5           # spatial CV splits
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_merged():
    print("Loading merged feature grid:", MERGED_GEO)
    g = gpd.read_file(MERGED_GEO)
    print("Grid cells:", len(g))
    return g

def prepare_features(grid):
    # select candidate features: VIIRS and tower-derived (present in your merged file)
    # adaptable: if column missing, skip
    viirs_cols = [c for c in ["nl_mean","nl_sum","nl_std","nl_count"] if c in grid.columns]
    tower_cols = [c for c in grid.columns if c.startswith("tower_") or c.startswith("mean_") or c.startswith("radio_") or c in ["tower_count","tower_density","nearest_tower_m","nl_per_tower"]]
    # ensure nl_per_tower if available
    all_feats = viirs_cols + tower_cols
    # dedupe
    all_feats = [c for i,c in enumerate(all_feats) if c not in all_feats[:i]]
    print("Using features ({}):".format(len(all_feats)), all_feats)
    X = grid[all_feats].fillna(0).copy()
    # Basic transforms: log-transform skewed columns where useful (nl_sum)
    if "nl_sum" in X.columns:
        X["nl_sum_log1p"] = np.log1p(X["nl_sum"].astype(float))
    return X, all_feats

def attach_target_from_raster(grid, raster_path):
    # Aggregate raster to grid by zonal stats using rasterstats (lazy import)
    print("Attaching target from raster:", raster_path)
    try:
        from rasterstats import zonal_stats
        import rasterio
    except Exception as e:
        raise RuntimeError("Please install rasterstats & rasterio to use raster ground truth. Error: " + str(e))
    # make sure CRS alignment
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    if grid.crs != raster_crs:
        grid = grid.to_crs(raster_crs)
    stats = zonal_stats(grid.geometry, raster_path, stats=["mean"], nodata=None, geojson_out=False)
    targ = pd.DataFrame(stats)["mean"].fillna(0)
    return targ, grid

def attach_target_from_csv(grid, csv_path):
    print("Attaching target from CSV:", csv_path)
    df = pd.read_csv(csv_path)
    # expect columns: grid_index or cell_id or geometry join key
    # try to find a matching id column
    if "cell_id" in df.columns:
        df = df.set_index("cell_id")
        # ensure grid has cell_id; if not, create index-based id
        if "cell_id" not in grid.columns:
            grid = grid.reset_index().rename(columns={"index":"cell_id"})
        targ = grid["cell_id"].map(df["target"]).fillna(0)
    elif "grid_index" in df.columns:
        if "index" not in grid.columns:
            grid = grid.reset_index().rename(columns={"index":"index"})
        targ = grid["index"].map(df.set_index("grid_index")["target"]).fillna(0)
    else:
        raise RuntimeError("CSV must have 'cell_id' or 'grid_index' to map to grid. Alternatively provide raster.")
    return targ, grid

def spatial_block_cv_splits(grid, n_splits=5, random_state=RANDOM_STATE):
    # Create spatial blocks using KMeans on centroids
    print("Creating spatial blocks for spatial CV (KMeans on centroids)...")
    centroids = np.vstack([grid.geometry.centroid.x, grid.geometry.centroid.y]).T
    # normalize coords
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    centroids_s = scaler.fit_transform(centroids)
    km = KMeans(n_clusters=n_splits, random_state=random_state)
    blocks = km.fit_predict(centroids_s)
    # GroupKFold with groups=blocks
    print("Block counts:", np.bincount(blocks))
    return blocks

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.nan
    return {"rmse": rmse, "mae": mae, "pearson_r": r}

def supervised_workflow(grid, X, y):
    print("Running supervised workflow (regression).")
    # create spatial groups
    groups = spatial_block_cv_splits(grid, n_splits=N_SPLITS)

    results = []
    # prepare feature sets (VIIRS-only vs Fused)
    viirs_cols = [c for c in X.columns if c.startswith("nl_") or c == "nl_sum_log1p"]
    fused_cols = X.columns.tolist()
    print("VIIRS-only features:", viirs_cols)
    print("Fused features:", fused_cols)

    # Modeling choices
    lgb_params = {"random_state": RANDOM_STATE, "n_jobs": -1}
    rf_params = {"random_state": RANDOM_STATE, "n_jobs": -1, "n_estimators": 200}

    # storage
    fold = 0
    preds_by_model = {}
    for model_name, feat_set in [("VIIRS_LGB", viirs_cols), ("FUSED_LGB", fused_cols), ("FUSED_RF", fused_cols)]:
        preds = np.zeros(len(X))
        for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
            fold += 1
            X_train, X_test = X.iloc[train_idx][feat_set], X.iloc[test_idx][feat_set]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # scale numeric features for RF? we do no scaling for tree models
            if model_name.endswith("_LGB"):
                dtrain = lgb.Dataset(X_train, label=y_train)
                model = lgb.train({}, dtrain, num_boost_round=200, verbose_eval=False)
                y_pred = model.predict(X_test)
            else:
                # RandomForest
                rf = RandomForestRegressor(**rf_params)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

            preds[test_idx] = y_pred

        metrics = evaluate_regression(y, preds)
        print(f"Model {model_name} metrics:", metrics)
        # save model trained on full data for explainability (only for LGB fused)
        if model_name == "FUSED_LGB":
            final = lgb.train({}, lgb.Dataset(X[feat_set], label=y), num_boost_round=300)
            joblib.dump(final, os.path.join(MODELS_DIR, "lgb_fused_full.pkl"))
            # SHAP
            explainer = shap.TreeExplainer(final)
            shap_values = explainer.shap_values(X[feat_set])
            # summary plot
            plt.figure(figsize=(8,6))
            shap.summary_plot(shap_values, X[feat_set], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_fused.png"), dpi=200)
            plt.close()
            # save feature importances
            fi = pd.DataFrame({"feature": feat_set, "importance": final.feature_importance()})
            fi = fi.sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(RESULTS_DIR, "feature_importances_fused.csv"), index=False)
        # store
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_performance.csv"), index=False)
    print("Saved model performance →", os.path.join(RESULTS_DIR, "model_performance.csv"))
    return results_df

def unsupervised_workflow(grid, X):
    print("Running unsupervised workflow: KMeans (k=3), HDBSCAN optional.")
    # VIIRS-only clustering
    viirs_cols = [c for c in X.columns if c.startswith("nl_") or c == "nl_sum_log1p"]
    km_viirs = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit_predict(X[viirs_cols])
    grid["cluster_viirs"] = km_viirs

    # fused clustering
    km_fused = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit_predict(X)
    grid["cluster_fused"] = km_fused

    # Save clusters
    grid.to_file(os.path.join(RESULTS_DIR, "clusters_viirs_vs_fused.geojson"), driver="GeoJSON")
    print("Saved cluster geojson:", os.path.join(RESULTS_DIR, "clusters_viirs_vs_fused.geojson"))

    # Basic cluster counts
    print("VIIRS cluster counts:", np.bincount(km_viirs))
    print("Fused cluster counts:", np.bincount(km_fused))

    # Optionally run HDBSCAN if installed
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
        grid["hdbscan_fused"] = clusterer.fit_predict(X)
        grid.to_file(os.path.join(RESULTS_DIR, "clusters_hdbscan_fused.geojson"), driver="GeoJSON")
        print("Saved HDBSCAN clusters.")
    except Exception:
        print("HDBSCAN not available or failed — skipping.")

    return grid

def main():
    grid = load_merged()
    X, feat_list = prepare_features(grid)

    if GROUND_TRUTH_RASTER or GROUND_TRUTH_CSV:
        # supervised
        if GROUND_TRUTH_RASTER:
            y, grid = attach_target_from_raster(grid, GROUND_TRUTH_RASTER)
        else:
            y, grid = attach_target_from_csv(grid, GROUND_TRUTH_CSV)
        # ensure X aligns with grid index (after potential reprojection)
        X = X.reindex(grid.index).fillna(0)
        results_df = supervised_workflow(grid, X, y)
        print("Supervised results:\n", results_df)
    else:
        # unsupervised path
        grid_out = unsupervised_workflow(grid, X)
        print("Unsupervised clustering saved. Inspect results in results/")

if __name__ == "__main__":
    main()
