import requests
import os
import csv

def download_mapillary_images(token, bbox, output_dir="mapillary_images", limit=10):

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "bbox": bbox,
        "fields": "id,geometry,thumb_1024_url,captured_at",
        "limit": limit
    }


    response = requests.get(url, params=params)
    data = response.json()


    with open(metadata_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "latitude", "longitude"])

        for i, image in enumerate(data.get('data', [])):
            img_url = image['thumb_1024_url']
            coords = image['geometry']['coordinates']  # [lon, lat]
            lat, lon = coords[1], coords[0]
            filename = f"image_{i+1}_{image['id']}.jpg"
            filepath = os.path.join(output_dir, filename)

            img_data = requests.get(img_url).content
            with open(filepath, "wb") as f:
                f.write(img_data)

            print(f"Downloaded: {filename} at lat: {lat}, lon: {lon}")
            writer.writerow([filename, lat, lon])


TOKEN = 'MLY|24037669172492017|b67c06b02b82afd4433addc6e06d7497'

bbox = '-122.520828,37.714702,-122.358340,37.825603'


os.makedirs("mapillary_images", exist_ok=True)

download_mapillary_images(
    token='MLY|24037669172492017|b67c06b02b82afd4433addc6e06d7497',
    bbox='-122.520828,37.714702,-122.358340,37.825603',
    output_dir='mapillary_images',
    limit=100
)