import torch
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset

from pyproj import Transformer

SF_LAT_MIN, SF_LAT_MAX = 37.7081, 37.830939
SF_LON_MIN, SF_LON_MAX = -122.5149, -122.383318
GRID_SIDE = 31

def get_sf_bounds_3857():
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    xmin, ymin = transformer.transform(SF_LON_MIN, SF_LAT_MIN)
    xmax, ymax = transformer.transform(SF_LON_MAX, SF_LAT_MAX)
    return xmin, xmax, ymin, ymax

def generate_grid_points(min_lon, min_lat, max_lon, max_lat, grid_side):
    lon_points = np.linspace(min_lon, max_lon, grid_side)
    lat_points = np.linspace(min_lat, max_lat, grid_side)
    grid = [(lon, lat) for lon in lon_points for lat in lat_points]
    return grid

def draw_prediction_probs(ax, coords, probs, color='red', alpha=0.4):
    for (lon, lat), p in zip(coords, probs):
        point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        ax.scatter(point.geometry.x, point.geometry.y, color=color, s=100 * p, alpha=alpha)

class TileClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes=961):
        super().__init__()
        self.encoder = clip_model.vision_model
        self.head = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.head(pooled_output)

def load_image_and_metadata(index, csv_path, image_dir, transform, processor):
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    image_path = os.path.join(image_dir, row['filename'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    if transform:
        image_pil = transform(image_pil)

    plt.imshow(image_pil)
    plt.title("Input Image")
    plt.axis("off")
    plt.savefig(f"Predictions_on_grid/{row['filename']}_image.png", dpi=300)
    plt.close()

    inputs = processor(images=image_pil, return_tensors="pt")
    return inputs['pixel_values'].squeeze(0), float(row['longitude']), float(row['latitude']), row['filename']

def visualize_prediction(index, model, processor, device, grid):
    transform = transforms.Compose([transforms.Resize((224, 224))])
    img_tensor, true_lon, true_lat, filename = load_image_and_metadata(
        index=index,
        csv_path="sf_mapillary_images/val_with_grid.csv",
        image_dir="sf_mapillary_images",
        transform=transform,
        processor=processor
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        topk = probs.argsort()[-6:][::-1]
        topk_probs = probs[topk]
        topk_coords = [grid[i] for i in topk]

    fig, ax = plt.subplots(figsize=(12, 10))

    true_point = gpd.GeoSeries([Point((true_lon, true_lat))], crs="EPSG:4326").to_crs(epsg=3857)
    ax.scatter(true_point.geometry.x, true_point.geometry.y, color="green", s=100, label="Ground Truth", edgecolor='black', zorder=1)

    draw_prediction_probs(ax, topk_coords, topk_probs, color="purple", alpha=0.6)

    xmin, xmax, ymin, ymax = get_sf_bounds_3857()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_title(f"Top-6 Predictions for Image: {filename}", fontsize=14)
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Predictions_on_grid/{filename}_predictions.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    model_path = "Gridmodels/_final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
    model = TileClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grid = generate_grid_points(SF_LON_MIN, SF_LAT_MIN, SF_LON_MAX, SF_LAT_MAX, GRID_SIDE)

    for idx in range(0, 20000, 200):
        visualize_prediction(idx, model, processor, device, grid)