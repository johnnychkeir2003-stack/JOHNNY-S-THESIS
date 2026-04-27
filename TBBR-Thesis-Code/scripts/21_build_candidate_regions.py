import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

# =========================
# CONFIG
# =========================
DATA_DIR = "raw_data/train/images"
LABELS_DIR = "working/yolo_labels/train"
OUTPUT_CSV = "working/candidate_regions/train_candidates.csv"

HEIGHT_PERCENTILE = 50
MIN_AREA = 30

# =========================
# HELPERS
# =========================
def load_yolo_boxes(label_path, img_shape):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    h, w = img_shape

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, bw, bh = map(float, parts)

            x_center *= w
            y_center *= h
            bw *= w
            bh *= h

            xmin = max(0, int(x_center - bw / 2))
            ymin = max(0, int(y_center - bh / 2))
            xmax = min(w - 1, int(x_center + bw / 2))
            ymax = min(h - 1, int(y_center + bh / 2))

            boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA >= xB or yA >= yB:
        return 0

    return (xB - xA) * (yB - yA)


def candidate_is_positive(candidate_box, gt_boxes):
    for gt in gt_boxes:
        if intersection_area(candidate_box, gt) > 0:
            return 1
    return 0

# =========================
# MAIN
# =========================
rows = []
images_processed = 0
images_with_labels = 0
total_gt_boxes = 0
total_positive_candidates = 0

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if not file.endswith(".npy"):
            continue

        npy_path = os.path.join(root, file)
        rel_path = os.path.relpath(npy_path, DATA_DIR).replace("\\", "/")
        block = rel_path.split("/")[0]

        img = np.load(npy_path)
        height = img[:, :, 4]
        h, w = height.shape

        # IMPORTANT FIX:
        # label name is block + "_" + stem + ".txt"
        stem = os.path.splitext(file)[0]
        label_name = f"{block}_{stem}.txt"
        label_path = os.path.join(LABELS_DIR, label_name)

        gt_boxes = load_yolo_boxes(label_path, (h, w))

        images_processed += 1
        if len(gt_boxes) > 0:
            images_with_labels += 1
            total_gt_boxes += len(gt_boxes)

        threshold = np.percentile(height, HEIGHT_PERCENTILE)
        mask = height >= threshold

        labeled = label(mask)
        regions = regionprops(labeled)

        for i, region in enumerate(regions):
            if region.area < MIN_AREA:
                continue

            min_row, min_col, max_row, max_col = region.bbox
            candidate_box = (min_col, min_row, max_col, max_row)

            label_val = candidate_is_positive(candidate_box, gt_boxes)
            if label_val == 1:
                total_positive_candidates += 1

            rows.append({
                "filename": file,
                "relative_path": rel_path,
                "block": block,
                "candidate_id": f"{rel_path}_{i}",
                "min_row": min_row,
                "min_col": min_col,
                "max_row": max_row,
                "max_col": max_col,
                "area": int(region.area),
                "label": int(label_val)
            })

        print(f"Processed: {rel_path} | Label file: {label_name} | GT boxes: {len(gt_boxes)}")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\nDONE")
print(f"Saved: {OUTPUT_CSV}")
print(f"Images processed: {images_processed}")
print(f"Images with labels found: {images_with_labels}")
print(f"Total GT boxes loaded: {total_gt_boxes}")
print(f"Total candidates: {len(df)}")
print(f"Unique candidate IDs: {df['candidate_id'].nunique()}")
print(f"Positive candidates: {int(df['label'].sum())}")
print(f"Negative candidates: {int((df['label'] == 0).sum())}")