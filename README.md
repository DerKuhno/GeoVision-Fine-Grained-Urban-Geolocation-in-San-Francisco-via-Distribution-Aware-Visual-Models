# 🗺️ GeoVision: Fine-Grained Urban Geolocation in San Francisco via Distribution-Aware Visual Models

This project explores fine-grained geolocation prediction within the city of San Francisco using image data and deep neural models. We build upon and extend the **StreetCLIP** model, using a **Mixture Density Network (MDN)** head and **Vision Transformer (ViT)** backbones to regress GPS coordinates from urban scene images. The system is benchmarked with and without data augmentation, and further visualized using attention rollout techniques.

## 📌 Project Highlights

- **Regression of GPS coordinates** using a distribution-aware Mixture Density Network (MDN) head.
- **Classifier prediction** using a classifier, we try to perdict grid points over San Franzisco.
- **StreetCLIP + ViT backbone**: Extending pretrained StreetCLIP with a custom ViT implementation.
- **Augmentation pipeline**: Includes affine transformations, blur, color jitter, and perspective warps.
- **Attention Rollout**: Visualize the spatial importance of different image regions for localization.
- **Dataset Tools**: Scripts for dataset cleaning, validation, and analysis of missing or corrupted files.

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `fine_tuning_with_freeze_using_clip.py` | Main training script for StreetCLIP regression with MDN |
| `fine_tuning_with_freeze.py` | Main training script for custom ViT regression with MDN |
| `fine_tuning_tiles_fast.py` | Main training script for Grid prediction |
| `attention_rollout_our_model.py` | Attention visualization on our custom ViT model |
| `attention_rollout.py` | Attention rollout using StreetCLIP’s ViT backbone |
| `griddataset.py`, `partial_datasets.py` | Dataset loaders and samplers for training and validation |
| `testin_on_vacation_photos.py` | Evaluate model on unseen (real-world) images |
| `visualize_*.py` | Various utilities to visualize predictions and attention |
| `check_dataset.py`, `check_images.py` | Dataset sanity checking, blank detection, and cleanup |
| `augmentation_sanitycheck.py` | View and debug the applied image augmentations |

## 🧪 Data

We use a curated subset of the **Mapillary dataset** focused on San Francisco. Each image has:
- GPS coordinates (latitude, longitude)
- Metadata for quality control
- Optional labels used for classification-style baselines

Preprocessing includes:
- Downsampling images
- Augmentation
- Normalization (StreetCLIP mean & std)

## 🖼️ Attention Visualization

To better understand model predictions, we implemented **attention rollout**:
- Shows how different patches in the image influence the final output.
- Used for both the baseline and custom ViT models.
- Plots are saved as heatmaps over the original images.


## 🎓 Course Context

This project was developed as part of **Stanford’s CS231n**. Our goal was to push the boundary of visual localization in dense urban environments using modern vision-language models.

## 📜 License

This project is for academic use only. For reproduction or use of the dataset/model, please refer to [Mapillary Terms] and the original [StreetCLIP] license.
