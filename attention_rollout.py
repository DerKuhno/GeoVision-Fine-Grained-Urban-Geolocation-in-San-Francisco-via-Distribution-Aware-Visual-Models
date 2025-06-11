import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fine_tuning_with_freeze_using_clip import StreetCLIPRegressor
from fine_tuning_tiles_fast import TileClassifier


class ViTAttentionRollout:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.encoder.layers:
            block.self_attn.register_forward_hook(self._get_attention)

    def _get_attention(self, module, input, output):
        attn_weights = output[1] 
        self.attentions.append(attn_weights.detach().cpu())

    def compute_rollout_attention(self):
        rollout = None
        for attn in self.attentions:
            attn_min = attn.min(dim=1).values
            attn_min += torch.eye(attn_min.size(-1))  
            attn_min /= attn_min.sum(dim=-1, keepdim=True)
            if rollout is None:
                rollout = attn_min
            else:
                rollout = rollout @ attn_min
        return rollout


def load_image(index, csv_path, image_dir, processor, transform):
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    image_path = os.path.join(image_dir, row["filename"])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    if transform:
        image_pil = transform(image_pil)
    inputs = processor(images=image_pil, return_tensors="pt")
    return image_pil, inputs["pixel_values"].squeeze(0), row["filename"]


def visualize_attention_map(image_pil, attn_map, filename):
    image_np = np.array(image_pil)
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode='bilinear', align_corners=False
    ).squeeze().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(attn_map, cmap='inferno', alpha=0.6)
    plt.title(f"Attention Rollout: {filename}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"attention no augmentation/{filename}")


if __name__ == "__main__":
    model_path = "logs3 (no augmentation)/Multigaus-no-augment-V3/_final.pt"
    #model_path = "logs1 (with augmentation)/Multigaus-augmentation-Finetuned-V1/_final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
    model = StreetCLIPRegressor(clip_model, K=6, batch_size=1, device=device).to(device)
    #model = TileClassifier(clip_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rollout = ViTAttentionRollout(model.encoder, device=device)

    transform = transforms.Compose([transforms.Resize((224, 224))])

    for index in range(0, 20000, 200):
        image_pil, image_tensor, filename = load_image(index, "sf_mapillary_images/val.csv", "sf_mapillary_images", processor, transform)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        rollout.attentions.clear()
        with torch.no_grad():
            _ = model(image_tensor)

        rollout_attn = rollout.compute_rollout_attention()[0]
        cls_to_patch_attn = rollout_attn[0, 1:] 
        num_patches = int(np.sqrt(cls_to_patch_attn.shape[0]))
        attn_map = cls_to_patch_attn.reshape(num_patches, num_patches)

        visualize_attention_map(image_pil, torch.tensor(attn_map), filename)