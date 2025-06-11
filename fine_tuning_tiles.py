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
import torch.nn.functional as F

class TileDataset(Dataset):
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
        label = int(row['grid_index'])
        return inputs['pixel_values'].squeeze(0), torch.tensor(label, dtype=torch.long)

def custom_collate_fn(batch):
    images, target_class = zip(*batch)
    images = torch.stack(images)
    target_class = torch.stack(target_class)
    return images, target_class

# 3. Model
class TileClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=961):
        super().__init__()
        self.encoder = clip_model.vision_model
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.head(pooled_output)


def class_loss(pred_classes, target_class):
    if target_class.ndim == 2 and target_class.shape[1] == 1:
        target_class = target_class.squeeze(1)

    return F.cross_entropy(pred_classes, target_class)


def train_model(model, dataloader, optimizer, device, loss_fn, scheduler):
    model.train()
    total_loss = 0

    scaler = GradScaler()

    for imgs, target_class in tqdm(dataloader, desc="Training", leave=False):
        imgs, target_class = imgs.to(device), target_class.to(device)
        
        optimizer.zero_grad()
        with autocast():
            pred_classes = model(imgs)
            loss = class_loss(pred_classes, target_class)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        scheduler.step()

    return total_loss / len(dataloader.dataset)


def pretrain_model(model, dataloader, optimizer, device, loss_fn, batch_size):
    model.train()
    total_loss = 0


    for step_count, (imgs, target_class) in enumerate(tqdm(dataloader, desc="Pretraining", leave=False)):
        imgs, target_class = imgs.to(device), target_class.to(device)    

        optimizer.zero_grad()

        pred_classes = model(imgs)
        loss = class_loss(pred_classes, target_class)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step_count%30==0 and step_count>0:
            break
    return total_loss / (step_count*batch_size)


def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, target_class in tqdm(dataloader, desc="Validation", leave=False):
            imgs, target_class = imgs.to(device), target_class.to(device)
            pred_classes = model(imgs)
            loss = class_loss(pred_classes, target_class)
            total_loss +=loss

    return total_loss/len(dataloader.dataset)


def preeval_model(model, dataloader, device, batch_size):
    model.eval()
    total_dist = 0
    step_count = 0
    total_loss = 0
    with torch.no_grad():
        for step_count, (imgs, target_class) in enumerate(tqdm(dataloader, desc="Prevalidation", leave=False)):
            imgs, target_class = imgs.to(device), target_class.to(device)
            pred_classes = model(imgs)
            loss = class_loss(pred_classes, target_class)
            total_loss +=loss

            if step_count%10==0 and step_count>0:
                break
    return total_loss/ (step_count*batch_size)

def run(save_model_name, freeze=False, epochs = 10, lr = 1e-4, batch_size = 16, threshhold=16, name_reuse_model=False, K=6):
    #config
    train_data_path = "sf_mapillary_images/train_with_grid.csv"
    val_data_path = "sf_mapillary_images/val_with_grid.csv"
    image_dir = "sf_mapillary_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    log_dir = os.path.join("logs", save_model_name)
    writer = SummaryWriter(log_dir=log_dir)


    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
    if freeze == False:

        clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
        model = TileClassifier(clip_model, num_classes=961).to(device)


        if name_reuse_model is not None:
            old_model = os.path.join("logs", name_reuse_model, f"_final.pt")
            model.load_state_dict(torch.load(old_model, map_location=device))
            print("Reused model: ", name_reuse_model)
        model = model.to(device)

    if freeze == True:
        clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP", trust_remote_code=True, use_safetensors=True)
        model = TileClassifier(clip_model, num_classes=961).to(device)

        for param in model.encoder.parameters():
            param.requires_grad = False


    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    train_dataset = TileDataset(train_data_path, image_dir, processor, transform)
    val_dataset = TileDataset(val_data_path, image_dir, processor, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8,
                            persistent_workers=True,
                            collate_fn=custom_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size*2,
                            num_workers=8, 
                            persistent_workers=True,
                            collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = class_loss
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if freeze==True:
            train_loss = pretrain_model(model, train_loader, optimizer, device, loss_fn, batch_size)
            val_loss = preeval_model(model, val_loader, device, batch_size)
            print("validation_dist = ", val_loss)
            break

        if freeze==False:
            train_loss = train_model(model, train_loader, optimizer, device, loss_fn, scheduler)
            torch.cuda.empty_cache()
            val_loss = eval_model(model, val_loader, device)
            print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.2f}")

        for param_group in optimizer.param_groups:
            print("Current LR:", param_group['lr'])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Error/train_km", train_loss, epoch)
        writer.add_scalar("Error/val_km", val_loss, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        if epoch%5 == 0:
            save_path = os.path.join("logs", save_model_name, f"_{epoch}_iter.pt")
            torch.save(model.state_dict(), save_path)


    save_path = os.path.join("logs", save_model_name, f"_final.pt")
    torch.save(model.state_dict(), save_path)
    writer.close()

if __name__ == "__main__":
    batch_size_freeze = 64
    lr_freeze = 1e-3
    epochs_freeze = 100
    freeze=True
    threshhold = 16 #about 10 miles
    save_model_name = "Classifier-V1"
    #run with freezed params
    run(save_model_name, freeze=freeze, epochs = epochs_freeze, lr = lr_freeze, batch_size = batch_size_freeze, threshhold=threshhold, name_reuse_model=False)

    #finetuning 
    batch_size = 22
    lr = 1e-4
    epochs = 100 #stops after the first 30steps of the first epoch
    freeze=False
    threshhold = None #about 10 miles
    save_model_name_fine = "Classifier-V1-Finetuned"
    run(save_model_name_fine, freeze=freeze, epochs = epochs, lr = lr, batch_size = batch_size, name_reuse_model=save_model_name)
