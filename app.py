# app.py — VIIRA: Research Analysis Dashboard
# Validated: Matches reference images + Fixed Density Normalization
# Run: python -m streamlit run app.py

import os
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ---------- CONFIG ----------
st.set_page_config(layout="wide", page_title="VIIRA Research Dashboard")

# 1. PATHS (Prioritize Distance file)
DIST_PATH = r"data/processed/merged_features_dist.geojson"
ORIG_PATH = r"data/processed/merged_features.geojson"

# 2. COLOR PALETTES
DECK_PALETTES = {
    "Inferno (Night Lights)": [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
    "Viridis (Infrastructure)": [[68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141], [42, 118, 142], [33, 144, 141], [31, 161, 135], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
    "RdYlGn (Distance)": [[165,0,38], [215,48,39], [244,109,67], [253,174,97], [254,224,139], [217,239,139], [166,217,106], [102,189,99], [26,152,80], [0,104,55]][::-1],
}

MAP_STYLES = {
    "Dark (Analysis)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Light (Context)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
}

# ---------- DATA LOADING ----------
@st.cache_data(ttl=600)
def load_data():
    path = DIST_PATH if os.path.exists(DIST_PATH) else ORIG_PATH
    if not os.path.exists(path):
        st.error(f"Data file not found. Please run scripts/calc_dist_fast.py first.")
        st.stop()
    
    gdf = gpd.read_file(path)
    if "cell_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "cell_id"})
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf

# ---------- NORMALIZATION LOGIC ----------
def robust_normalize(series, method="minmax", p_min=2, p_max=98):
    s = series.fillna(0)
    
    # FIX: Density/Counts must use Log1p to handle 0 vs 1 vs 100 correctly
    if method == "Logarithmic":
        s_log = np.log1p(s)
        min_v, max_v = s_log.min(), s_log.max()
        if max_v == min_v: return s_log * 0
        return (s_log - min_v) / (max_v - min_v)
        
    elif method == "Percentile Clip":
        lower, upper = np.percentile(s, p_min), np.percentile(s, p_max)
        s_clipped = s.clip(lower, upper)
        denom = upper - lower
        if denom == 0: return s_clipped * 0
        return (s_clipped - lower) / denom
        
    else: # Linear
        min_v, max_v = s.min(), s.max()
        if max_v == min_v: return s * 0
        return (s - min_v) / (max_v - min_v)

# ---------- PLOT HELPERS ----------
def get_deck_layer(gdf, col, scaling, palette_name, opacity, reverse=False):
    palette = DECK_PALETTES.get(palette_name, DECK_PALETTES["Viridis (Infrastructure)"])
    if reverse: palette = palette[::-1]

    # Normalize data for the map
    gdf["norm_val"] = robust_normalize(gdf[col], method=scaling)
    
    def get_color(row):
        val_norm = row["norm_val"]
        if pd.isna(val_norm): return [0,0,0,0]
        
        idx = int(val_norm * (len(palette) - 1))
        idx = max(0, min(idx, len(palette) - 1))
        return palette[idx] + [int(opacity * 255)]

    gdf["fill_color"] = gdf.apply(get_color, axis=1)

    return pdk.Layer(
        "GeoJsonLayer", data=gdf, get_fill_color="fill_color",
        get_line_color=[0,0,0,0], pickable=True, stroked=False, filled=True
    ), f"<b>Cell ID:</b> {{cell_id}}<br/><b>{col}:</b> {{{col}}}"

def generate_static_plot(gdf, col, scaling, cmap, bg_color="#0e1117", reverse_cmap=False):
    """
    Generates high-contrast static figures matching your reference images.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Background Setup
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    text_color = 'white' if bg_color in ['black', '#000000', '#0e1117'] else 'black'
    
    # 2. Land Silhouette (Crucial for "Empty" areas to be visible)
    # If black background -> Dark Grey Land (#262730)
    # If white background -> Light Grey Land (#e0e0e0)
    land_fill = '#262730' if text_color == 'white' else '#e0e0e0'
    gdf.plot(ax=ax, color=land_fill, edgecolor=None)

    # 3. Data Plotting
    plot_data = robust_normalize(gdf[col], method=scaling)
    gdf['temp_plot'] = plot_data
    
    # Handle Matplotlib Reversal
    if reverse_cmap and not cmap.endswith("_r"): cmap += "_r"
    elif not reverse_cmap and cmap.endswith("_r"): cmap = cmap[:-2]
    
    # Mask 0 values so the land silhouette shows through for empty data
    # This prevents "Zero" looking like "Lowest Value Color"
    gdf_plot = gdf[gdf['temp_plot'] > 0] if "dist" not in col else gdf # Don't mask distance (0 is good!)
    
    if not gdf_plot.empty:
        gdf_plot.plot(column='temp_plot', cmap=cmap, legend=True, ax=ax,
                 legend_kwds={'label': f"Normalized {col}", 'orientation': "horizontal", 'pad': 0.05})
    
    # 4. Styling
    try:
        cbar = ax.get_figure().axes[-1]
        cbar.tick_params(colors=text_color, labelsize=10)
        cbar.xaxis.label.set_color(text_color)
        cbar.outline.set_edgecolor(text_color)
    except: pass
    
    ax.set_axis_off()
    ax.set_title(f"Spatial Distribution of {col}", color=text_color, fontsize=16, pad=20)
    return fig

# ---------- MAIN UI ----------
st.sidebar.title("VIIRA Research Suite")
gdf = load_data()
numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()

# NAVIGATION
view_mode = st.sidebar.radio("Analysis Mode", ["Exploratory Map", "Comparative (Split View)", "Statistical Analysis"])

# =========================================================
# VIEW 1: EXPLORATORY MAP (For Figure 1 & 2)
# =========================================================
if view_mode == "Exploratory Map":
    st.header("Exploratory Spatial Analysis")
    col1, col2 = st.columns([1, 3])
    with col1:
        target_col = st.selectbox("Feature", numeric_cols, index=0)
        
        # --- SMART DEFAULTS ---
        is_dist = "dist" in target_col or "km" in target_col
        is_density = "density" in target_col or "count" in target_col or "num" in target_col
        
        st.subheader("Settings")
        
        # Scaling Logic: Distance=Linear, Density=Log, Lights=Percentile
        default_scale = 0 # Percentile
        if is_dist: default_scale = 2 # Linear
        if is_density: default_scale = 1 # Logarithmic
        
        scaling = st.radio("Scaling", ["Percentile Clip", "Logarithmic", "Linear"], index=default_scale)
        
        # Palette Logic
        default_pal = 0 # Inferno
        if is_dist: default_pal = 2 # RdYlGn
        if is_density: default_pal = 1 # Viridis
        
        palette = st.selectbox("Palette", list(DECK_PALETTES.keys()), index=default_pal)
        
        reverse = st.checkbox("Reverse Colors", value=False, help="Use for Distance (Green=0km)")
        opacity = st.slider("Opacity", 0.0, 1.0, 0.9)
        map_style = st.selectbox("Base Map", list(MAP_STYLES.keys()))

    with col2:
        layer, tooltip_html = get_deck_layer(gdf, target_col, scaling, palette, opacity, reverse)
        view_state = pdk.ViewState(latitude=gdf.geometry.centroid.y.mean(), longitude=gdf.geometry.centroid.x.mean(), zoom=8.5)
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=MAP_STYLES[map_style], tooltip={"html": tooltip_html})
        st.pydeck_chart(deck)
    
    # --- EXPORT SECTION ---
    st.markdown("---")
    st.subheader("📷 Export Figure")
    e1, e2 = st.columns(2)
    with e1: 
        # Crucial: "Pure Black" option for your reference look
        export_bg = st.radio("Background", ["Pure Black (#000000)", "Streamlit Dark (#0e1117)", "White"], index=0, horizontal=True)
    
    # Map selection to hex codes
    bg_hex = "#000000"
    if "Streamlit" in export_bg: bg_hex = "#0e1117"
    if "White" in export_bg: bg_hex = "white"
    
    # Matplotlib Cmap Mapping
    cmap_lookup = {"Inferno": "inferno", "Viridis": "viridis", "RdYlGn": "RdYlGn"}
    export_cmap = cmap_lookup.get(palette.split(" ")[0], "viridis")
    
    fig = generate_static_plot(gdf, target_col, scaling, export_cmap, bg_hex, reverse_cmap=reverse)
    st.pyplot(fig)
    
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=bg_hex)
    st.download_button("Download High-Res PNG", buf.getvalue(), f"figure_{target_col}.png", "image/png")

# =========================================================
# VIEW 2: COMPARATIVE (For Figure 4)
# =========================================================
elif view_mode == "Comparative (Split View)":
    st.header("Comparative Analysis")
    c1, c2 = st.columns(2)
    with c1: col_left = st.selectbox("Left Map", numeric_cols, index=0)
    with c2: col_right = st.selectbox("Right Map", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
    
    # Settings specifically for the Split View export
    split_bg = st.radio("Background", ["Pure Black", "White"], index=0, horizontal=True)
    bg_hex = "#000000" if split_bg == "Pure Black" else "white"
    text_color = "white" if split_bg == "Pure Black" else "black"
    land_fill = "#262730" if split_bg == "Pure Black" else "#e0e0e0"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor(bg_hex)
    
    # LEFT MAP
    gdf.plot(ax=ax1, color=land_fill, edgecolor=None)
    gdf['temp_l'] = robust_normalize(gdf[col_left], "Percentile Clip") # Default smart scaling
    gdf.plot(column='temp_l', ax=ax1, cmap='inferno', legend=True, legend_kwds={'shrink': 0.5})
    ax1.set_title(f"A: {col_left}", color=text_color, fontsize=18)
    ax1.set_axis_off()
    
    # RIGHT MAP
    gdf.plot(ax=ax2, color=land_fill, edgecolor=None)
    # Smart scaling for right map (check if distance)
    scale_r = "Linear" if ("dist" in col_right or "km" in col_right) else "Percentile Clip"
    gdf['temp_r'] = robust_normalize(gdf[col_right], scale_r)
    
    # Smart Palette for right map
    cmap_r = 'RdYlGn_r' if ("dist" in col_right or "km" in col_right) else 'viridis'
    gdf.plot(column='temp_r', ax=ax2, cmap=cmap_r, legend=True, legend_kwds={'shrink': 0.5})
    ax2.set_title(f"B: {col_right}", color=text_color, fontsize=18)
    ax2.set_axis_off()
    
    st.pyplot(fig)
    
    # Download Button for Split View
    buf_split = BytesIO()
    fig.savefig(buf_split, format="png", dpi=300, bbox_inches='tight', facecolor=bg_hex)
    st.download_button("Download Comparative PNG", buf_split.getvalue(), "comparative_view.png", "image/png")

# =========================================================
# VIEW 3: STATISTICAL (For Scatter Plot)
# =========================================================
elif view_mode == "Statistical Analysis":
    st.header("Statistical Validation")
    
    # 1. SCATTER PLOT
    st.subheader("1. Bivariate Scatter Plot")
    s1, s2 = st.columns(2)
    with s1: x_axis = st.selectbox("X-Axis (e.g. Lights)", numeric_cols, index=0)
    with s2: y_axis = st.selectbox("Y-Axis (e.g. Distance)", numeric_cols, index=1 if len(numeric_cols)>1 else 0)

    # Clean Data Controls
    with st.expander("🛠️ Graph Settings", expanded=True):
        col_clean, col_log = st.columns(2)
        with col_clean:
            remove_outliers = st.checkbox("Remove Top 1% Outliers", value=True)
        with col_log:
            use_log_x = st.checkbox("Log Scale (X-Axis)", value=False)
            use_log_y = st.checkbox("Log Scale (Y-Axis)", value=False)
    
    # Filter Logic
    plot_df = gdf.copy()
    if remove_outliers:
        p99_x = plot_df[x_axis].quantile(0.99)
        p99_y = plot_df[y_axis].quantile(0.99)
        plot_df = plot_df[ (plot_df[x_axis] <= p99_x) & (plot_df[y_axis] <= p99_y) ]

    # Plot
    fig_scat, ax_scat = plt.subplots(figsize=(8, 6))

    if use_log_x:
        ax_scat.set_xscale("log")

    if use_log_y:
        ax_scat.set_yscale("log")

    sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, alpha=0.3, s=15, color="teal", ax=ax_scat, edgecolor=None)
    
    # Trendline (Only if linear)
    if not use_log_x:
        sns.regplot(data=plot_df, x=x_axis, y=y_axis, scatter=False, color="red", ax=ax_scat, line_kws={"linewidth": 2})

    ax_scat.set_title(f"{x_axis} vs {y_axis} (Cleaned)" if remove_outliers else f"{x_axis} vs {y_axis} (Raw)")
    ax_scat.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig_scat)
    
    # Download Button for Scatter
    buf_scat = BytesIO()
    fig_scat.savefig(buf_scat, format="png", dpi=300, bbox_inches='tight')
    st.download_button("Download Scatter Plot", buf_scat.getvalue(), "scatter_plot.png", "image/png")

    # 2. CORRELATION
    st.subheader("2. Correlation Matrix")
    if len(numeric_cols) > 1:
        corr = gdf[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)