import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import os
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2



class PreNormTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
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
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class CustomViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=1024, depth=12, heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
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
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * patch_H * patch_W)

        x = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)

        return type('Output', (object,), {"pooler_output": x[:, 0]})()


class GeoDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        lat, lon = float(row['latitude']), float(row['longitude'])

        return inputs['pixel_values'].squeeze(0), torch.tensor([lat, lon], dtype=torch.float32)


def custom_collate_fn(batch):
    images, coords = zip(*batch)
    images = torch.stack(images)
    coords = torch.stack(coords)
    return images, coords

class StreetCLIPRegressor(nn.Module):
    def __init__(self, clip_model, K, batch_size, device):
        super().__init__()
        self.K = K
        self.encoder = clip_model.vision_model

        self.coords_mean = torch.tensor([37.7695, -122.4491], dtype=torch.float32, device=device).unsqueeze(0)
        self.coords_scale = torch.tensor([16.28, 15.2], dtype=torch.float32, device=device).unsqueeze(0)
        # self.projector = clip_model.visual_projection
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
        outputs = self.encoder(pixel_values=pixel_values, output_attentions=True)
        pooled_output = outputs.pooler_output
        h = self.head(pooled_output)
        pi = nn.functional.softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(-1, self.K, 2)
        sigma = torch.exp(self.sigma(h))
        return pi, mu, sigma
    

    def shifting_coords(self, coords):
        coords = coords - self.coords_mean
        return coords * self.coords_scale

    def shifting_back(self, coords):
        coords = coords/self.coords_scale
        return coords + self.coords_mean


def mdn_loss(pi, mu, sigma, target):
    B, K, _ = mu.shape
    target = target.unsqueeze(1).expand_as(mu)
    var = (sigma ** 2).unsqueeze(2).expand_as(mu)
    coeff = 1.0 / (2 * np.pi * (sigma+ 1e-6))

    exp_term = -0.5 * torch.sum(((target - mu) ** 2) / var, dim=2)
    weighted_gauss = pi * coeff * torch.exp(exp_term)
    probs = torch.sum(weighted_gauss, dim=1)
    nll = -torch.log(probs + 1e-9)

    return torch.mean(nll)

def mdn_argmax(pi, mu):

    _, max_indices = torch.max(pi, dim=1) 
    argmax = torch.stack([
        mu[i, max_indices[i]] for i in range(mu.size(0))
    ], dim=0)

    return argmax


def haversine(a, b):
    R = 6371  #earth radius
    lat1, lon1 = a[:, 0], a[:, 1]
    lat2, lon2 = b[:, 0], b[:, 1]
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    lat1 = torch.deg2rad(lat1)
    lat2 = torch.deg2rad(lat2)
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    return R * c

def train_model(model, dataloader, optimizer, device, loss_fn, scheduler):
    model.train()
    total_loss = 0
    total_dist = 0

    scaler = GradScaler()

    for imgs, targets in tqdm(dataloader, desc="Training", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        targets = model.shifting_coords(targets)
        
        optimizer.zero_grad()
        with autocast():
            pi, mu, sigma = model(imgs)
            loss = mdn_loss(pi, mu, sigma, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #loss.backward()
        #optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            preds = model.shifting_back(mdn_argmax(pi, mu))
            targets = model.shifting_back(targets)
            dist = haversine(preds, targets)
        total_dist += dist.sum().item()
        scheduler.step()
    return total_loss / len(dataloader), total_dist / len(dataloader.dataset)


def pretrain_model(model, dataloader, optimizer, device, loss_fn, batch_size):
    model.train()
    total_loss = 0
    total_dist = 0
    step_count = 0


    for step_count, (imgs, targets) in enumerate(tqdm(dataloader, desc="Pretraining", leave=False)):
        imgs, targets = imgs.to(device), targets.to(device)    
        targets = model.shifting_coords(targets)

        optimizer.zero_grad()

        pi, mu, sigma = model(imgs)
        loss = mdn_loss(pi, mu, sigma, targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            preds = model.shifting_back(mdn_argmax(pi, mu))
            targets = model.shifting_back(targets)
            dist = haversine(preds, targets)
        total_dist += dist.sum().item()
        if step_count%30==0 and step_count>0:
            break
    return total_loss / (step_count*batch_size), total_dist / (step_count*batch_size)


def eval_model(model, dataloader, device):
    model.eval()
    total_dist = 0
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Validation", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            pi, mu, _ = model(imgs)
            preds = model.shifting_back(mdn_argmax(pi, mu))
            dist = haversine(preds, targets)
            total_dist += dist.sum().item()
    return total_dist / len(dataloader.dataset)

def preeval_model(model, dataloader, device, batch_size):
    model.eval()
    total_dist = 0
    step_count = 0
    with torch.no_grad():
        for step_count, (imgs, targets) in enumerate(tqdm(dataloader, desc="Prevalidation", leave=False)):
            imgs, targets = imgs.to(device), targets.to(device)
            pi, mu, _ = model(imgs)
            preds = model.shifting_back(mdn_argmax(pi, mu))
            dist = haversine(preds, targets)
            total_dist += dist.sum().item()
            if step_count%10==0 and step_count>0:
                break
    return total_dist / (step_count*batch_size)

def run(save_model_name, freeze=False, epochs = 10, lr = 1e-4, batch_size = 16, threshhold=16, name_reuse_model=False, K=6):
    #config
    train_data_path = "sf_mapillary_images/train.csv"
    val_data_path = "sf_mapillary_images/val.csv"
    image_dir = "sf_mapillary_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join("logs3", save_model_name)
    writer = SummaryWriter(log_dir=log_dir)


    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
    if freeze == False:
        clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
        model = StreetCLIPRegressor(clip_model, K, batch_size, device).to(device)

        if name_reuse_model is not None:
            old_model = os.path.join("logs3", name_reuse_model, f"_final.pt")
            model.load_state_dict(torch.load(old_model, map_location=device))
            print("Reused model: ", name_reuse_model)
        model = model.to(device)

    if freeze == True:
        clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
        model = StreetCLIPRegressor(clip_model, K, batch_size, device).to(device)
        for param in model.encoder.parameters():
            param.requires_grad = False


    # train_transform = transforms.Compose([
    # transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))], p=0.8),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),
    # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),
    # transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    # transforms.Resize((224, 224)),
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    train_dataset = GeoDataset(train_data_path, image_dir, processor, transform)
    val_dataset = GeoDataset(val_data_path, image_dir, processor, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8,
                            persistent_workers=True,
                            collate_fn=custom_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size*2,
                            num_workers=8,
                            persistent_workers=True,
                            collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if freeze==True:
            train_loss, train_dist = pretrain_model(model, train_loader, optimizer, device, loss_fn, batch_size)
            val_dist = preeval_model(model, val_loader, device, batch_size)
            print("validation_dist = ", val_dist)
            if val_dist < threshhold:
                print(f"Threshhold achieved at epoch {epoch} with Val Error (km) {val_dist}")
                break

        if freeze==False:
            train_loss, train_dist = train_model(model, train_loader, optimizer, device, loss_fn, scheduler)
            torch.cuda.empty_cache()
            val_dist = eval_model(model, val_loader, device)
            print(f"[{epoch+1}/{epochs}] Train Error (km): {train_dist:.4f} | Val Error (km): {val_dist:.2f}")

        for param_group in optimizer.param_groups:
            print("Current LR:", param_group['lr'])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Error/train_km", train_dist, epoch)
        writer.add_scalar("Error/val_km", val_dist, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        if epoch&6==0:
            save_path = os.path.join("logs3", save_model_name, f"_{epoch}_iter.pt")
            torch.save(model.state_dict(), save_path)


    if (freeze == True) and (val_dist >= threshhold):
        print("threshhold not achieved")

    save_path = os.path.join("logs3", save_model_name, f"_final.pt")
    torch.save(model.state_dict(), save_path)
    writer.close()

if __name__ == "__main__":
    batch_size_freeze = 64
    lr_freeze = 1e-4
    epochs_freeze = 100
    freeze=True
    threshhold = 16 #about 10 miles
    save_model_name = "Multigaus-no-augment-V3"
    #run with freezed params
    run(save_model_name, freeze=freeze, epochs = epochs_freeze, lr = lr_freeze, batch_size = batch_size_freeze, threshhold=threshhold, name_reuse_model=False)

    #finetuning 
    batch_size = 22
    lr = 5e-5
    epochs = 25
    freeze=False
    threshhold = None #about 10 miles
    save_model_name_fine = "Multigaus-no-augment-V3"
    run(save_model_name_fine, freeze=freeze, epochs = epochs, lr = lr, batch_size = batch_size, name_reuse_model=save_model_name)
