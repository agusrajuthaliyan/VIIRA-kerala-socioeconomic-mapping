# VIIRA: Research Analysis Dashboard

VIIRA (VIIRS Infrastructure & Integration Research Analysis) is a geospatial research dashboard designed for analyzing the relationship between Night-Time Lights (VIIRS) and telecommunications infrastructure (OpenCellID) in Kerala.

The dashboard provides interactive tools for exploratory spatial analysis, comparative visualization, and statistical validation, helping researchers understand urbanization patterns and infrastructure distribution.

## 🚀 Features

*   **Exploratory Spatial Analysis**: Interactive 3D maps using PyDeck to visualize spatial distributions of various features (Night Lights, Tower Density, Distance metrics).
*   **Comparative Analysis**: Split-view mode to visually compare two different features side-by-side with synchronized extents.
*   **Statistical Validation**: Integrated scatter plots and correlation matrices to analyze relationships between variables quantitatively.
*   **High-Quality Exports**: Tools to generate and download publication-ready figures (maps and charts) with customizable high-contrast themes (Pure Black, Dark, White).
*   **Advanced Scaling**: Robust data normalization options including Logarithmic, Percentile Clipping, and Linear scaling to handle skewed geospatial data.

## 🏙️ City Visualizations

Here are a few high-resolution map renders generated for various cities in Kerala:

<p align="center">
  <img src="Resources/Cities/Kerala.png" width="48%" alt="Kerala" />
  <img src="Resources/Cities/Thrissur-emarald.png" width="48%" alt="Thrissur Emerald" />
</p>
<p align="center">
  <img src="Resources/Cities/kochi-neon.png" width="48%" alt="Kochi Neon" />
  <img src="Resources/Cities/ernakulam-noir.png" width="48%" alt="Ernakulam Noir" />
</p>
<p align="center">
  <img src="Resources/Cities/kozhikode-rustic.png" width="48%" alt="Kozhikode Rustic" />
  <img src="Resources/Cities/trivandrum-blueprint.png" width="48%" alt="Trivandrum Blueprint" />
</p>


## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/agusrajuthaliyan/VIIRA-kerala-socioeconomic-mapping.git
    cd VIIRA
    ```

2.  **Set up a virtual environment (Recommended)**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Data Setup

The project relies on a processed GeoJSON dataset. The data processing pipeline is located in the `scripts/` directory.

1.  **Raw Data**: Ensure raw data source files (VIIRS .tif files, etc.) are in the `data/raw/` directory (if applicable for your setup).
2.  **Run Pipeline**: Scripts `01` through `06` process the raw data into a merged feature set.
3.  **Compute Distances**:
    The dashboard prioritizes `data/processed/merged_features_dist.geojson`. If this file is missing, you must generate it:
    ```bash
    python scripts/calc_dist_fast.py
    ```
    *Note: The app will throw an error if this file is not found.*

## 🖥️ Usage

To launch the interactive dashboard:

```bash
python -m streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

### Dashboard Modes:
*   **Exploratory Map**: Primary view for visualizing single variables. Use the sidebar to select features, adjust scaling, and change color palettes (Inferno, Viridis, etc.).
*   **Comparative (Split View)**: Select distinct variables for the Left and Right maps to compare spatial patterns directly.
*   **Statistical Analysis**: Analyze bivariate relationships with scatter plots (supports log scales and outlier removal) and view correlation heatmaps.

## 📂 Project Structure

```
VIIRA/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── data/                       # Data directory
│   ├── processed/              # Generated GeoJSON files
│   └── raw/                    # Source datasets
├── scripts/                    # Data processing pipeline
│   ├── 01_extract_kerala...    # Boundary extraction
│   ├── ...
│   ├── 07_model_training.py    # ML model training
│   ├── 08_model_shap.py        # SHAP value analysis
│   └── calc_dist_fast.py       # Distance computation utility
└── results/                    # Output directory for figures and models
```

## 🤝 Contributing
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License
[MIT License](LICENSE)
