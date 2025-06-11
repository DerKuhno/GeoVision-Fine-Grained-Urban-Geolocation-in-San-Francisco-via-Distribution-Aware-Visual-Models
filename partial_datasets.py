import pandas as pd
import os
import random

def split_metadata(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df_shuffled)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")


    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) to {output_dir}")

def validate_splits(output_dir):
    train_filenames = pd.read_csv(os.path.join(output_dir, "train.csv"))['filename']
    val_filenames = pd.read_csv(os.path.join(output_dir, "val.csv"))['filename']
    test_filenames = pd.read_csv(os.path.join(output_dir, "test.csv"))['filename']

    train_set = set(train_filenames)
    val_set = set(val_filenames)
    test_set = set(test_filenames)
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if not overlap_train_val and not overlap_train_test and not overlap_val_test:
        print("no overlap between train, validation, and test sets.")
    else:
        print("overlap detected:")
        if overlap_train_val:
            print(f"- Train & Val overlap: {len(overlap_train_val)} items")
        if overlap_train_test:
            print(f"- Train & Test overlap: {len(overlap_train_test)} items")
        if overlap_val_test:
            print(f"- Val & Test overlap: {len(overlap_val_test)} items")

if __name__ == "__main__":
    original_csv = os.path.join("sf_mapillary_images", "metadata.csv")
    output_dir = "sf_mapillary_images"

    if os.path.exists(original_csv):
        split_metadata(original_csv, output_dir)
        validate_splits(output_dir)