import geopandas as gpd
import matplotlib.pyplot as plt

# Paths
GRID_PATH = "data/processed/merged_features.geojson"
TOWER_PATH = "data/processed/opencellid_kerala.geojson"
OUT_PATH = "paper/figures/fig5_nl_tower_overlay.png"

# Load data
grid = gpd.read_file(GRID_PATH)
towers = gpd.read_file(TOWER_PATH)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Nightlight base
grid.plot(
    column="nl_mean",
    cmap="inferno",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Nighttime Light Mean"}
)

# Tower overlay
towers.plot(
    ax=ax,
    color="cyan",
    markersize=2,
    alpha=0.6
)

ax.set_title("Nighttime Light Intensity with Telecom Tower Locations (Kerala)")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Saved Figure 5 to {OUT_PATH}")
