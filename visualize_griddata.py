import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import numpy as np
import os


MIN_LON, MIN_LAT = -122.5149, 37.7081
MAX_LON, MAX_LAT = -122.383318, 37.830939
GRID_SIDE = 31

def generate_grid_points(min_lon, min_lat, max_lon, max_lat, grid_side):
    lon_points = np.linspace(min_lon, max_lon, grid_side)
    lat_points = np.linspace(min_lat, max_lat, grid_side)
    grid = [(lon, lat) for lon in lon_points for lat in lat_points]
    return grid

def visualize_grid_image_distribution(csv_path):
    df = pd.read_csv(csv_path)
    counts = df.groupby('grid_index').size().reset_index(name='image_count')


    grid = generate_grid_points(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, GRID_SIDE)
    grid_df = pd.DataFrame(grid, columns=['lon', 'lat'])
    grid_df['grid_index'] = grid_df.index
    grid_df = grid_df.merge(counts, on='grid_index', how='left')
    grid_df['image_count'] = grid_df['image_count'].fillna(0)


    grid_df['geometry'] = grid_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gdf = gpd.GeoDataFrame(grid_df, geometry='geometry', crs='EPSG:4326')
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_web.plot(ax=ax, column='image_count', cmap='plasma', legend=True, markersize=60, alpha=0.8)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_title("San Francisco: Image Count Per Grid Point", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("grid_image_distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    input_csv = os.path.join("sf_mapillary_images", "metadata_with_grid.csv")
    if os.path.exists(input_csv):
        visualize_grid_image_distribution(input_csv)