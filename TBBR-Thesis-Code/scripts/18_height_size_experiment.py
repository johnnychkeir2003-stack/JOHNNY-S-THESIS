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

OUTPUT_DIR = os.path.join(BASE_DIR, "working", "height_size_experiment")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, f"{SPLIT}_height_size_experiment.csv")

# -------------------------
# PARAMETERS
# -------------------------
HEIGHT_PERCENTILES = [40, 50, 60, 70]
MIN_COMPONENT_AREAS = [10, 20, 30, 50, 80]

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
    mask = (height_channel >= h_thr).astype(np.uint8) * 255

    # optional cleanup to remove isolated pixels
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, h_thr


def build_annotation_mask(img_height, img_width, anns):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

    return mask


def connected_components(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    for label_id in range(1, n_labels):  # skip background
        area = stats[label_id, cv2.CC_STAT_AREA]
        comp = (labels == label_id).astype(np.uint8) * 255
        comps.append((label_id, comp, area))

    return comps


def filter_by_size(comps, min_area):
    filtered = []

    for _, comp_mask, area in comps:
        if area < min_area:
            continue

        filtered.append({
            "mask": comp_mask,
            "area": int(area),
        })

    return filtered


def overlaps(mask_a, mask_b):
    overlap = cv2.bitwise_and(mask_a, mask_b)
    return np.count_nonzero(overlap) > 0


def annotation_retention(anns, filtered_comps, img_height, img_width):
    retained = 0
    missed = 0

    for ann in anns:
        ann_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(ann_mask, [pts], 255)

        found_overlap = False
        for comp in filtered_comps:
            if overlaps(comp["mask"], ann_mask):
                found_overlap = True
                break

        if found_overlap:
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
    for min_area in MIN_COMPONENT_AREAS:
        total_annotations = 0
        total_retained = 0
        total_missed = 0
        total_components_before = 0
        total_components_after = 0

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
            comps = connected_components(height_mask)
            filtered_comps = filter_by_size(comps, min_area)

            retained, missed = annotation_retention(anns, filtered_comps, img_height, img_width)

            total_annotations += len(anns)
            total_retained += retained
            total_missed += missed
            total_components_before += len(comps)
            total_components_after += len(filtered_comps)

        results.append({
            "height_percentile": height_percentile,
            "min_component_area": min_area,
            "images_used": len(images),
            "components_before_size_filter": total_components_before,
            "components_after_size_filter": total_components_after,
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
print("\n=== HEIGHT + SIZE EXPERIMENT RESULTS ===")
for row in results:
    print(
        f"Height {row['height_percentile']}% | "
        f"Min area {row['min_component_area']} | "
        f"Retention: {row['annotations_retained']}/{row['total_annotations']} "
        f"({row['annotation_retention_percent']}%) | "
        f"Components after size filter: {row['components_after_size_filter']}"
    )

print("\nCSV saved to:", CSV_PATH)
