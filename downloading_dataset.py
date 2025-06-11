import requests
import os
import csv
import numpy as np

def generate_grid(min_lon, min_lat, max_lon, max_lat, step=0.005):

    lon_points = np.arange(min_lon, max_lon, step)
    lat_points = np.arange(min_lat, max_lat, step)
    grid = [(lon, lat) for lon in lon_points for lat in lat_points]
    return grid

def load_seen_ids_from_csv(csv_path):
    seen_ids = set()

    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 1:
                filename = row[0] 
                img_id = filename.rsplit('.', 1)[0] 
                seen_ids.add(img_id)
            if row%1000:
                print(f"added {row} images to the set")
    return seen_ids

def download_images_around_point(token, center_lon, center_lat, delta=0.001, limit=10, seen_ids=set(), output_dir="sf_mapillary_images", csv_writer=None, downloaded_count=[0], max_total_images=100000):
    if downloaded_count[0] >= max_total_images:
        return

    bbox = f"{center_lon - delta},{center_lat - delta},{center_lon + delta},{center_lat + delta}"
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "bbox": bbox,
        "fields": "id,geometry,thumb_1024_url",
        "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for idx, image in enumerate(data.get('data', [])):
            if downloaded_count[0] >= max_total_images:
                return
            if idx >= limit:
                break

            img_id = image['id']
            if img_id in seen_ids:
                continue 
            seen_ids.add(img_id)

            coords = image['geometry']['coordinates']  # [lon, lat]
            lat, lon = coords[1], coords[0]
            img_url = image['thumb_1024_url']
            filename = f"{img_id}.jpg"
            filepath = os.path.join(output_dir, filename)

            img_data = requests.get(img_url).content
            with open(filepath, "wb") as f:
                f.write(img_data)

            csv_writer.writerow([filename, lat, lon])

            file_size = len(img_data)
            downloaded_count[1] += file_size/1000 #in KB

            downloaded_count[0] += 1
            if downloaded_count[0] % 100 == 0:
                print(f"Downloaded {downloaded_count[0]} images...")
                print(f"With Memory of {downloaded_count[1]*64/1000} MB or {downloaded_count[1]*64/1e+6} GB")

    except Exception as e:
        print(f"Error at point {center_lat}, {center_lon}: {e}")
        
def download_sanfrancisco_grid(token):
    min_lon, min_lat = -122.5149, 37.7081 
    max_lon, max_lat = -122.383318, 37.830939 

    output_dir = "sf_mapillary_images"
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    seen_ids = set()
    downloaded_count = [0, 0]
    file_exists = os.path.isfile(metadata_path)

    with open(metadata_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["filename", "latitude", "longitude"]) 

        if file_exists:

            print("reusing the csv")

        step = (max_lat-min_lat)/100

        grid = generate_grid(min_lon, min_lat, max_lon, max_lat, step)

        for center_lon, center_lat in grid:
            if downloaded_count[0] >= 100000:
                print("Reached target of 100,000 images.")
                return
            download_images_around_point(
                token=token,
                center_lon=center_lon,
                center_lat=center_lat,
                delta=step/2,
                limit=10,
                seen_ids=seen_ids,
                output_dir=output_dir,
                csv_writer=writer,
                downloaded_count=downloaded_count,
                max_total_images=100000,
            )


TOKEN = 'MLY|24037669172492017|b67c06b02b82afd4433addc6e06d7497'

number_images = 10
download_sanfrancisco_grid(TOKEN)
