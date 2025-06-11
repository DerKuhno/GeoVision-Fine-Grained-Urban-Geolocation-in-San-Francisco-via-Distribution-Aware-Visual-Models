import csv
import numpy as np
import os
from scipy.spatial import KDTree

MIN_LON, MIN_LAT = -122.5149, 37.7081
MAX_LON, MAX_LAT = -122.383318, 37.830939

def generate_grid(min_lon, min_lat, max_lon, max_lat, num_points=1000):
    grid_side = int(np.sqrt(num_points))
    lon_points = np.linspace(min_lon, max_lon, grid_side)
    lat_points = np.linspace(min_lat, max_lat, grid_side)
    grid = [(lon, lat) for lon in lon_points for lat in lat_points]
    return grid

def find_nearest_grid_index(grid, lat, lon, kdtree=None):
    if kdtree is None:
        kdtree = KDTree(grid)
    _, idx = kdtree.query((lon, lat))
    return idx

def assign_nearest_grid_points(csv_path, output_path, grid):
    print(len(grid))
    with open(csv_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        
        rows = list(reader)

    grid_array = np.array(grid)
    kdtree = KDTree(grid_array)

    updated_rows = []
    for row in rows:
        filename, lat, lon = row[0], float(row[1]), float(row[2])
        idx = find_nearest_grid_index(grid, lat, lon, kdtree)
        updated_rows.append([filename, lat, lon, idx])

    with open(output_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["filename", "latitude", "longitude", "grid_index"])
        writer.writerows(updated_rows)

    print(f"Saved output to {output_path}. Total rows: {len(updated_rows)}")

if __name__ == "__main__":
    input_csv = "sf_mapillary_images/metadata.csv"
    output_csv = "sf_mapillary_images/metadata_with_grid.csv"

    grid = generate_grid(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, num_points=1000)
    assign_nearest_grid_points(input_csv, output_csv, grid)



