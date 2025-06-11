import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import seaborn as sns
import os
import numpy as np

def visualize_on_satellite(csv_path):
    df = pd.read_csv(csv_path)
    df['lat_rounded'] = df['latitude'].round(4)
    df['lon_rounded'] = df['longitude'].round(4)

    grouped = df.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
    grouped['geometry'] = grouped.apply(lambda row: Point(row['lon_rounded'], row['lat_rounded']), axis=1)

    gdf = gpd.GeoDataFrame(grouped, geometry='geometry', crs='EPSG:4326')
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_web_mercator.plot(ax=ax, column='count', cmap='viridis', legend=True, markersize=50, alpha=0.8)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_title("San Francisco: Image Count per Location (Satellite Background)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("satellite_overlay.png", dpi=300)
    plt.show()

    all_points = pd.DataFrame()
    all_points["geometry"] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    all_points = gpd.GeoDataFrame(all_points, geometry='geometry', crs='EPSG:4326')
    all_points = all_points.to_crs(epsg=3857)
    coords = np.array([[p.x, p.y] for p in all_points.geometry])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.kdeplot(
        x=coords[:, 0],
        y=coords[:, 1],
        fill=True,
        cmap="Reds",
        alpha=0.7,
        bw_adjust=0.05,
        ax=ax,
        levels=100,
        thresh=0
    )
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_title("San Francisco: Image Density Heatmap (Satellite Background)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("satellite_heatmap.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    metadata_csv_path = os.path.join("sf_mapillary_images", "metadata.csv")
    if os.path.exists(metadata_csv_path):
        visualize_on_satellite(metadata_csv_path)