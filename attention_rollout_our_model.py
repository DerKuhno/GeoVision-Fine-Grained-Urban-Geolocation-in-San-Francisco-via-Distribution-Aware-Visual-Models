import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class PreNormTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.save_attn = False
        self.attn_weights = None

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        if self.save_attn:
            self.attn_weights = attn_weights.detach().cpu()
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class CustomViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=1024, depth=12, heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            PreNormTransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        patch_H, patch_W = self.patch_size, self.patch_size
        patches = pixel_values.unfold(2, patch_H, patch_H).unfold(3, patch_W, patch_W)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, C * patch_H * patch_W)
        x = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)  # ✅ Use transformer directly

        x = self.norm(x)
        return type('Output', (object,), {"pooler_output": x[:, 0]})()

class StreetCLIPRegressor(nn.Module):
    def __init__(self, clip_model, K, batch_size, device):
        super().__init__()
        self.K = K
        self.encoder = clip_model
        self.coords_mean = torch.tensor([37.7695, -122.4491], dtype=torch.float32, device=device).unsqueeze(0)
        self.coords_scale = torch.tensor([16.28, 15.2], dtype=torch.float32, device=device).unsqueeze(0)

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.pi = nn.Linear(256, self.K)
        self.mu = nn.Linear(256, self.K * 2)
        self.sigma = nn.Linear(256, self.K)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        h = self.head(pooled_output)
        pi = nn.functional.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.K, 2)
        sigma = torch.exp(self.sigma(h))
        return pi, mu, sigma

    def shifting_back(self, coords):
        return coords / self.coords_scale + self.coords_mean

class AttentionRolloutCustomViT:
    def __init__(self, model):
        self.model = model
        self.attn_maps = []

    def enable_attention_tracking(self):
        for block in self.model.transformer:
            block.save_attn = True

    def collect_attention_maps(self):
        self.attn_maps = [block.attn_weights for block in self.model.transformer if block.attn_weights is not None]

    def compute_rollout_attention(self):
        rollout = None
        for attn in self.attn_maps:
            attn = attn.mean(dim=1)
            attn = attn + torch.eye(attn.size(-1), device=attn.device).unsqueeze(0) 
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn if rollout is None else torch.bmm(rollout, attn)
        return rollout 

def visualize_attention_on_image(image, attn_map, filename="attn_map"):
    image_np = np.array(image)
    attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=image_np.shape[:2], mode='bilinear').squeeze()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(attn_map.cpu(), cmap='inferno', alpha=0.5)
    plt.title(f"Attention Rollout: {filename}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"attention for ours/{filename}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "logs2 (our model with augementation)/Multigaus-Our-model-augmentation-Finetuned-V1/_final.pt"

    model = StreetCLIPRegressor(CustomViT(), K=6, batch_size=1, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for index in range(0, 20000, 200):
        csv_path = "sf_mapillary_images/val.csv"
        image_dir = "sf_mapillary_images"
        df = pd.read_csv(csv_path)
        row = df.iloc[index]
        filename = row["filename"]
        image_path = os.path.join(image_dir, filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image).resize((224, 224))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        rollout = AttentionRolloutCustomViT(model.encoder)
        rollout.enable_attention_tracking()

        with torch.no_grad():
            _ = model(image_tensor)

        rollout.collect_attention_maps()
        attn = rollout.compute_rollout_attention()[0]
        print("attn.shape", attn.shape)
        cls_attn = attn[0, 1:] 
        grid = 224 // model.encoder.patch_size
        attn_map = cls_attn.reshape(grid, grid)


        visualize_attention_on_image(image_pil, attn_map, filename=filename)