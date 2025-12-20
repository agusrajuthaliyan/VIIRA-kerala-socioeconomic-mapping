import geopandas as gpd
import folium

g = gpd.read_file("data/processed/merged_features.geojson")

# Create ID column for Folium
g = g.reset_index().rename(columns={'index':'cell_id'})

# Create map centered on Kerala
m = folium.Map(location=[10.2,76.4], zoom_start=8)

# Add choropleth
folium.Choropleth(
    geo_data=g.to_json(),
    data=g,
    columns=['cell_id','nl_mean'],
    key_on='feature.properties.cell_id',
    fill_color='YlOrRd',
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name="Nightlights Mean"
).add_to(m)

# Save
m.save("data/outputs/maps/viirs_choropleth.html")

print("Map saved! Open: data/outputs/maps/viirs_choropleth.html")
