import os
import numpy as np
import pandas as pd

DATA_DIR = "raw_data/test/images"
CANDIDATE_CSV = "working/candidate_regions/test_candidates.csv"
OUTPUT_CSV = "working/candidate_features/test_candidate_features_rgb.csv"

df = pd.read_csv(CANDIDATE_CSV)
df = df.drop_duplicates(subset=["candidate_id"])

rows = []

for rel_path in df["relative_path"].unique():
    npy_path = os.path.join(DATA_DIR, rel_path.replace("/", os.sep))

    if not os.path.exists(npy_path):
        print(f"Missing: {npy_path}")
        continue

    img = np.load(npy_path)

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    thermal = img[:, :, 3]
    height = img[:, :, 4]

    sub_df = df[df["relative_path"] == rel_path]

    for _, row in sub_df.iterrows():
        min_row = int(row["min_row"])
        min_col = int(row["min_col"])
        max_row = int(row["max_row"])
        max_col = int(row["max_col"])

        b_crop = blue[min_row:max_row, min_col:max_col]
        g_crop = green[min_row:max_row, min_col:max_col]
        r_crop = red[min_row:max_row, min_col:max_col]
        thermal_crop = thermal[min_row:max_row, min_col:max_col]
        height_crop = height[min_row:max_row, min_col:max_col]

        if thermal_crop.size == 0:
            continue

        rows.append({
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "block": row["block"],
            "candidate_id": row["candidate_id"],
            "label": int(row["label"]),

            "blue_mean": float(np.mean(b_crop)),
            "blue_std": float(np.std(b_crop)),
            "green_mean": float(np.mean(g_crop)),
            "green_std": float(np.std(g_crop)),
            "red_mean": float(np.mean(r_crop)),
            "red_std": float(np.std(r_crop)),

            "thermal_mean": float(np.mean(thermal_crop)),
            "thermal_max": float(np.max(thermal_crop)),
            "thermal_min": float(np.min(thermal_crop)),
            "thermal_std": float(np.std(thermal_crop)),

            "height_mean": float(np.mean(height_crop)),
            "height_max": float(np.max(height_crop)),
            "height_min": float(np.min(height_crop)),
            "height_std": float(np.std(height_crop)),

            "area": row["area"],
            "bbox_width": max_col - min_col,
            "bbox_height": max_row - min_row,
            "aspect_ratio": (max_col - min_col) / (max_row - min_row + 1e-6)
        })

    print(f"Processed test features: {rel_path}")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

features_df = pd.DataFrame(rows)
features_df.to_csv(OUTPUT_CSV, index=False)

print("\nDONE")
print(f"Saved: {OUTPUT_CSV}")
print(f"Total feature rows: {len(features_df)}")
print(f"Unique candidate IDs: {features_df['candidate_id'].nunique()}")
print(f"Positive count: {features_df['label'].sum()}")
print(f"Negative count: {(features_df['label'] == 0).sum()}")
