import os
import pandas as pd
from PIL import Image, ImageStat
import numpy as np
import matplotlib.pyplot as plt


def review_images(report_csv, image_dir, filter_condition):

    df = pd.read_csv(report_csv)
    if filter_condition not in df.columns:
        print(f"Column '{filter_condition}' not found in report.")
        return

    subset = df[df[filter_condition]]
    print(f"Found {len(subset)} images with condition: {filter_condition}")

    for idx, row in subset.iterrows():
        img_path = os.path.join(image_dir, row["filename"])
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"{row['filename']}\n[Press close or 'q' to skip, 'd' to delete]")
            plt.axis('off')
            plt.show()

            while True:
                action = input("Delete this image? [d=delete / q=skip / x=exit]: ").strip().lower()
                if action == 'd':
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")
                    break
                elif action == 'q':
                    print("Skipped.")
                    break
                elif action == 'x':
                    print("Exiting...")
                    return
                else:
                    print("Invalid input. Press 'd' to delete, 'q' to skip, 'x' to exit.")

        except Exception as e:
            print(f"Error reading {img_path}: {e}")

def update_csvs_after_deletion(csv_dir, image_dir, csv_files=["train.csv", "val.csv", "test.csv", "metadata.csv"]):

    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f" File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            print(f" 'filename' column missing in {csv_file}")
            continue

        original_len = len(df)
        df["exists"] = df["filename"].apply(lambda fn: os.path.exists(os.path.join(image_dir, fn)))
        df = df[df["exists"]].drop(columns=["exists"])
        new_len = len(df)

        df.to_csv(csv_path, index=False)
        print(f"✅ {csv_file}: {original_len - new_len} entries removed, {new_len} remaining.")

update_csvs_after_deletion(
    csv_dir="sf_mapillary_images",
    image_dir="sf_mapillary_images"
)

review_images(
    report_csv="sf_mapillary_images/image_quality_report.csv",
    image_dir="sf_mapillary_images",
    filter_condition="is_blank"
)


IMAGE_DIR = "sf_mapillary_images"
CSV_PATH = os.path.join(IMAGE_DIR, "metadata.csv")
REPORT_PATH = os.path.join(IMAGE_DIR, "image_quality_report.csv")


MIN_AREA = 100 * 100  
MAX_AREA = 5000 * 5000 
MIN_MEAN_BRIGHTNESS = 5
MAX_MEAN_BRIGHTNESS = 250 
BLANKNESS_THRESHOLD = 10  


# df = pd.read_csv(CSV_PATH)
# filenames = df["filename"].tolist()


# results = []
# for fname in filenames:
#     img_path = os.path.join(IMAGE_DIR, fname)
#     stats = {
#         "filename": fname,
#         "exists": False,
#         "readable": False,
#         "width": None,
#         "height": None,
#         "mean_pixel": None,
#         "is_too_small": False,
#         "is_too_large": False,
#         "is_too_dark": False,
#         "is_too_bright": False,
#         "is_blank": False,
#         "error": None
#     }

#     if not os.path.exists(img_path):
#         stats["error"] = "File not found"
#         results.append(stats)
#         continue

#     stats["exists"] = True
#     try:
#         with Image.open(img_path) as img:
#             img = img.convert("L")  # convert to grayscale
#             width, height = img.size
#             stats["width"] = width
#             stats["height"] = height
#             area = width * height

#             # Size checks
#             stats["is_too_small"] = area < MIN_AREA
#             stats["is_too_large"] = area > MAX_AREA

#             # Brightness check
#             mean_pixel = ImageStat.Stat(img).mean[0]
#             stats["mean_pixel"] = mean_pixel
#             stats["is_too_dark"] = mean_pixel < MIN_MEAN_BRIGHTNESS
#             stats["is_too_bright"] = mean_pixel > MAX_MEAN_BRIGHTNESS
#             stats["is_blank"] = mean_pixel < BLANKNESS_THRESHOLD

#             stats["readable"] = True
#     except Exception as e:
#         stats["error"] = str(e)

#     results.append(stats)

# report_df = pd.DataFrame(results)
# report_df.to_csv(REPORT_PATH, index=False)
# print(f"Image quality report saved to: {REPORT_PATH}")


# print("\n--- SUMMARY ---")
# print("Unreadable images:", len(report_df[~report_df['readable']]))
# print("Too small:", len(report_df[report_df['is_too_small']]))
# print("Too large:", len(report_df[report_df['is_too_large']]))
# print("Too dark:", len(report_df[report_df['is_too_dark']]))
# print("Too bright:", len(report_df[report_df['is_too_bright']]))
# print("Blank images:", len(report_df[report_df['is_blank']]))