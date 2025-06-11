import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import geopandas as gpd
import contextily as ctx
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from shapely.geometry import Point
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from fine_tuning_with_freeze_using_clip import mdn_argmax#, StreetCLIPRegressor
from fine_tuning_with_freeze import CustomViT, StreetCLIPRegressor, GeoDataset
from pyproj import Transformer

SF_LAT_MIN, SF_LAT_MAX = 37.70, 37.82
SF_LON_MIN, SF_LON_MAX = -122.52, -122.35

def get_sf_bounds_3857():
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    xmin, ymin = transformer.transform(SF_LON_MIN, SF_LAT_MIN)
    xmax, ymax = transformer.transform(SF_LON_MAX, SF_LAT_MAX)
    return xmin, xmax, ymin, ymax

def draw_gaussian_mixture(ax, means, stds, weights, color="blue", alpha=0.3):
    for i in range(len(means)):
        mean = means[i]
        std = stds[i]
        weight = weights[i]
        print(weights[i])
        print(stds[i])
        ellipse = Ellipse(
            xy=mean,
            width=std * 1000,
            height=std * 1000,
            angle=0,
            alpha=alpha,# * weight,
            color=color,
            linewidth=1.5
        )
        ax.add_patch(ellipse)


def load_image_and_metadata(index, csv_path, image_dir, transform, processor):
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    image_path = os.path.join(image_dir, row["filename"])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    if transform:
        image_pil = transform(image_pil)
    
    plt.imshow(image_pil)
    plt.title("Transformed Input Image")
    plt.axis("off")
    plt.savefig(f"predictions_with_ours/{row['filename']}_image.png", dpi=300)
    plt.close()

    inputs = processor(images=image_pil, return_tensors="pt")
    gt_coords = torch.tensor([float(row["latitude"]), float(row["longitude"])])
    return inputs["pixel_values"].squeeze(0), gt_coords, row["filename"]

def load_image_and_metadata_our_model(index, csv_path, image_dir):
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    image_path = os.path.join(image_dir, row["filename"])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_save = Image.fromarray(image)
    image_pil = Image.fromarray(image).resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    plt.imshow(image_save)
    plt.title("Transformed Input Image")
    plt.axis("off")
    #plt.savefig(f"predictions_with_ours/{row['filename']}_image.png", dpi=300)
    plt.show()
    plt.close()

    gt_coords = torch.tensor([float(row["latitude"]), float(row["longitude"])])
    return image_tensor.squeeze(0), gt_coords, row["filename"]

def visualize_prediction(index, model, device):
    transform = transforms.Compose([transforms.Resize((224, 224))])
    # img_tensor, gt_coords, filename = load_image_and_metadata(
    #    index=index,
    #    csv_path="sf_mapillary_images/val.csv",
    #    image_dir="sf_mapillary_images",
    #    transform=transform,
    #    processor=processor
    # )
    img_tensor, gt_coords, filename = load_image_and_metadata_our_model(
        index=index,
        csv_path="sf_mapillary_images/val.csv",
        image_dir="sf_mapillary_images",
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pi, mu, sigma = model(img_tensor)
        mu = model.shifting_back(mu.squeeze(0)).cpu().numpy()
        pi = pi.squeeze(0).cpu().numpy()
        sigma = sigma.squeeze(0).cpu().numpy()
        gt_coords = gt_coords.numpy()
        pi_tensor = torch.tensor(pi, device=device).unsqueeze(0)
        mu_tensor = torch.tensor(mu, device=device).unsqueeze(0)
        pred_coords = mdn_argmax(pi_tensor, mu_tensor).squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))


    gt_point = gpd.GeoSeries([Point(gt_coords[1], gt_coords[0])], crs="EPSG:4326").to_crs(epsg=3857)
    pred_points = gpd.GeoSeries([Point(lon, lat) for lat, lon in mu], crs="EPSG:4326").to_crs(epsg=3857)
    pred_coords_3857 = np.array([[p.x, p.y] for p in pred_points])
    gt_coords_3857 = np.array([[p.x, p.y] for p in gt_point])
    draw_gaussian_mixture(ax, pred_coords_3857, sigma, pi, color="purple", alpha=0.4)
    ax.scatter(gt_coords_3857[:, 0], gt_coords_3857[:, 1], color="green", s=100, label="Ground Truth")

    pred_point = gpd.GeoSeries([Point(pred_coords[1], pred_coords[0])], crs="EPSG:4326").to_crs(epsg=3857)
    #ax.scatter(pred_point.geometry.x, pred_point.geometry.y, color="red", s=80, label="Prediction (argmax)")

    xmin, xmax, ymin, ymax = get_sf_bounds_3857()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_title(f"Prediction for Image: {filename}", fontsize=14)
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    #plt.savefig(f"predictions_with_ours/{filename}_predictions.png", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    model_path="logs2 (our model with augementation)/Multigaus-Our-model-augmentation-Finetuned-V1/_final.pt"
    #model_path="logs3 (no augmentation)/Multigaus-no-augment-V3/_final.pt"
    print("Exists:", os.path.exists(model_path))
    print("Size:", os.path.getsize(model_path) if os.path.exists(model_path) else "File not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
    clip_model = CustomViT()
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)

    #model = StreetCLIPRegressor(clip_model, K=6, batch_size=1, device=device).to(device)
    model = StreetCLIPRegressor(clip_model, K=6, batch_size=1, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for idx in range(0, 20000, 200):
        visualize_prediction(idx, model, device)


    # plt.imshow(image_pil)
    # plt.title("Transformed Input Image")
    # plt.axis("off")
    # plt.show()
    # plt.close()