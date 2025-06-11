import os
import pandas as pd

def check_images_in_directory(csv_path, image_dir):

    df = pd.read_csv(csv_path)
    filenames = df['filename'].tolist()

    missing_files = []
    for fname in filenames:
        full_path = os.path.join(image_dir, fname)
        if not os.path.isfile(full_path):
            missing_files.append(fname)

    print(f"Checked {len(filenames)} images.")
    print(f" Found: {len(filenames) - len(missing_files)}")
    print(f" Missing: {len(missing_files)}")

    if missing_files:
        print("Missing filenames:")
        for fname in missing_files:
            print(" -", fname)

    return missing_files

csv_path = "/home/ubuntu/project/sf_mapillary_images/metadata.csv"
image_dir = "/home/ubuntu/project/sf_mapillary_images"
print("HI")
missing = check_images_in_directory(csv_path, image_dir)