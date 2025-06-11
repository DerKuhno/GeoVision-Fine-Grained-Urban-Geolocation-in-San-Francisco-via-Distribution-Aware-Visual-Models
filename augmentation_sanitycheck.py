import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


image_dir = "sf_mapillary_images"
csv_file = "sf_mapillary_images/metadata.csv"
example_image = "100720632861403.jpg"


import pandas as pd
df = pd.read_csv(csv_file)
if len(df) > 0:
    example_image = os.path.join(image_dir, df.iloc[0]['filename'])


extreme_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ], p=0.8),
    transforms.RandomApply([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    ], p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.Resize((224, 224)),
])

resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def show_augmented_examples(img_path, transform, resize_transform, num_examples=5):
    img = load_image(img_path)

    base_img = resize_transform(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(base_img)
    plt.axis('off')
    plt.title("Original (Resized Only)")
    plt.tight_layout()
    plt.show()

    for i in range(num_examples):
        torch.manual_seed(i+42)
        aug_img = transform(img)
        plt.figure(figsize=(4, 4))
        plt.imshow(aug_img)
        plt.axis('off')
        plt.title(f"Example Augmentation")
        plt.tight_layout()
        plt.show()


if example_image:
    print(f"Loaded image: {example_image}")
    show_augmented_examples(example_image, extreme_transform, resize_transform, num_examples=5)
