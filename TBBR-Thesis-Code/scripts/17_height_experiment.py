import json
import os
import cv2
import numpy as np
import csv

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

# -------------------------
# CONFIG
# -------------------------
SPLIT = "test"   # "train" or "test"
MAX_IMAGES = 30  # use first N images

TRAIN_JSON = os.path.join(BASE_DIR, "raw_data", "train", "Flug1_100-104Media_coco.json")
TEST_JSON = os.path.join(BASE_DIR, "raw_data", "test", "Flug1_105Media_coco.json")

TRAIN_RAW = os.path.join(BASE_DIR, "raw_data", "train", "images")
TEST_RAW = os.path.join(BASE_DIR, "raw_data", "test", "images")

OUTPUT_DIR = os.path.join(BASE_DIR, "working", "height_experiment")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, f"{SPLIT}_height_experiment.csv")

# Height thresholds to test
HEIGHT_PERCENTILES = [40, 50, 60, 70]

# -------------------------
# HELPERS
# -------------------------
def load_config():
    if SPLIT == "train":
        return TRAIN_JSON, TRAIN_RAW
    elif SPLIT == "test":
        return TEST_JSON, TEST_RAW
    else:
        raise ValueError("SPLIT must be 'train' or 'test'")


def file_name_to_raw_path(raw_base, file_name):
    parts = file_name.replace("\\", "/").split("/")
    block = parts[-2]
    fname = parts[-1]
    return os.path.join(raw_base, block, fname)


def build_height_mask(height_channel, height_percentile):
    h_thr = np.percentile(height_channel, height_percentile)
    height_mask = (height_channel >= h_thr).astype(np.uint8) * 255
    return height_mask, h_thr


def build_annotation_mask(height, width, anns):
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

    return mask


def overlaps(mask_a, mask_b):
    overlap = cv2.bitwise_and(mask_a, mask_b)
    return np.count_nonzero(overlap) > 0


def annotation_retention_with_height_only(anns, height_mask, img_height, img_width):
    retained = 0
    missed = 0

    for ann in anns:
        ann_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(ann_mask, [pts], 255)

        if overlaps(ann_mask, height_mask):
            retained += 1
        else:
            missed += 1

    return retained, missed


# -------------------------
# MAIN
# -------------------------
json_path, raw_base = load_config()

with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"][:MAX_IMAGES]
annotations = coco["annotations"]

ann_by_image = {}
for ann in annotations:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

results = []

for height_percentile in HEIGHT_PERCENTILES:
    total_annotations = 0
    total_retained = 0
    total_missed = 0

    for img_info in images:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        img_height = img_info["height"]
        img_width = img_info["width"]

        raw_path = file_name_to_raw_path(raw_base, file_name)
        if not os.path.exists(raw_path):
            print("Missing raw file:", raw_path)
            continue

        arr = np.load(raw_path)
        height_channel = arr[:, :, 4]

        anns = ann_by_image.get(image_id, [])
        if len(anns) == 0:
            continue

        height_mask, h_thr = build_height_mask(height_channel, height_percentile)
        retained, missed = annotation_retention_with_height_only(
            anns, height_mask, img_height, img_width
        )

        total_annotations += len(anns)
        total_retained += retained
        total_missed += missed

    results.append({
        "height_percentile": height_percentile,
        "images_used": len(images),
        "total_annotations": total_annotations,
        "annotations_retained": total_retained,
        "annotations_missed": total_missed,
        "annotation_retention_percent": round(
            100 * total_retained / total_annotations, 2
        ) if total_annotations > 0 else 0.0,
    })

# Save CSV
fieldnames = list(results[0].keys()) if results else []
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# Print summary
print("\n=== HEIGHT-ONLY EXPERIMENT RESULTS ===")
for row in results:
    print(
        f"Height percentile {row['height_percentile']}% | "
        f"Annotations retained: {row['annotations_retained']}/{row['total_annotations']} "
        f"({row['annotation_retention_percent']}%)"
    )

print("\nCSV saved to:", CSV_PATH)