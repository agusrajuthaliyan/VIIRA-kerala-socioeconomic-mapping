# scripts/06_model_shap.py

import geopandas as gpd
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# ---------- PATHS ----------
DATA_PATH = "data/processed/merged_features_dist.geojson"
OUT_DIR = "results/shap"
MODEL_DIR = "results/model"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
gdf = gpd.read_file(DATA_PATH)

features = [
    "dist_to_tower_km",
    "tower_density",
    "mean_range",
    "mean_avgsignal",
    "area_km2"
]

X = gdf[features].fillna(0)
y = gdf["nl_mean"]

# ---------- TRAIN MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
joblib.dump(model, f"{MODEL_DIR}/rf_nl_model.pkl")

# ---------- SHAP ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=300)
plt.close()

# Dependence plot (KEY FIGURE)
plt.figure()
shap.dependence_plot(
    "dist_to_tower_km",
    shap_values,
    X_test,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_dist_dependence.png", dpi=300)
plt.close()

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

ablations = {
    "Full Model": [
        "nl_mean", "nl_std",
        "tower_density", "dist_to_tower_km",
        "mean_range", "mean_avgsignal",
        "area_km2"
    ],
    "Nightlights Only": ["nl_mean", "nl_std"],
    "Infrastructure Only": [
        "tower_density", "dist_to_tower_km",
        "mean_range", "mean_avgsignal"
    ],
    "Nightlights + Distance": [
        "nl_mean", "nl_std", "dist_to_tower_km"
    ]
}

results = []

for name, feats in ablations.items():
    X = gdf[feats].fillna(0)
    y = gdf["nl_mean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    })

df_ablation = pd.DataFrame(results)
df_ablation.to_csv("results/ablation_results.csv", index=False)
print(df_ablation)
