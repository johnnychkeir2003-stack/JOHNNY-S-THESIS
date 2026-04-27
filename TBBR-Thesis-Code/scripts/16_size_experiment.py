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

OUTPUT_DIR = os.path.join(BASE_DIR, "working", "size_experiment")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, f"{SPLIT}_size_experiment.csv")

# -------------------------
# FIXED PARAMETERS
# -------------------------
THERMAL_PERCENTILE = 95
HEIGHT_PERCENTILE = 60
MIN_ELONGATION = 2.0

# Only size changes
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


def build_annotation_mask(height, width, anns):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
    return mask


def build_candidate_mask(thermal, height_channel):
    h_thr = np.percentile(height_channel, HEIGHT_PERCENTILE)
    building_mask = (height_channel >= h_thr)

    if np.any(building_mask):
        t_thr = np.percentile(thermal[building_mask], THERMAL_PERCENTILE)
    else:
        t_thr = np.percentile(thermal, THERMAL_PERCENTILE)

    hot_mask = (thermal >= t_thr)
    candidate_mask = (building_mask & hot_mask).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    return candidate_mask


def component_masks(binary_mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    comps = []
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        comp = (labels == label_id).astype(np.uint8) * 255
        comps.append((label_id, comp, area))
    return comps


def filter_components(comps, min_area):
    filtered = []

    for _, comp_mask, area in comps:
        ys, xs = np.where(comp_mask > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        if min(width, height) == 0:
            continue

        elongation = max(width, height) / min(width, height)

        if area < min_area:
            continue

        if elongation < MIN_ELONGATION:
            continue

        filtered.append({
            "mask": comp_mask,
            "area": int(area),
            "elongation": float(elongation),
        })

    return filtered


def overlaps(mask_a, mask_b):
    overlap = cv2.bitwise_and(mask_a, mask_b)
    return np.count_nonzero(overlap) > 0


def annotation_retention(anns, filtered_comps, height, width):
    retained = 0
    missed = 0

    for ann in anns:
        ann_mask = np.zeros((height, width), dtype=np.uint8)
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

for min_area in MIN_COMPONENT_AREAS:
    total_candidates_before = 0
    total_candidates_after = 0
    total_covered = 0
    total_uncovered = 0
    total_annotations = 0
    total_annotations_retained = 0
    total_annotations_missed = 0

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
        thermal = arr[:, :, 3]
        height_channel = arr[:, :, 4]

        anns = ann_by_image.get(image_id, [])
        ann_mask = build_annotation_mask(img_height, img_width, anns)

        candidate_mask = build_candidate_mask(thermal, height_channel)
        comps = component_masks(candidate_mask)
        filtered_comps = filter_components(comps, min_area)

        total_candidates_before += len(comps)
        total_candidates_after += len(filtered_comps)

        covered = 0
        uncovered = 0

        for comp in filtered_comps:
            if overlaps(comp["mask"], ann_mask):
                covered += 1
            else:
                uncovered += 1

        total_covered += covered
        total_uncovered += uncovered

        retained, missed = annotation_retention(anns, filtered_comps, img_height, img_width)

        total_annotations += len(anns)
        total_annotations_retained += retained
        total_annotations_missed += missed

    results.append({
        "min_component_area": min_area,
        "images_used": len(images),
        "candidate_hotspots_before_filter": total_candidates_before,
        "candidate_hotspots_after_filter": total_candidates_after,
        "covered_candidates": total_covered,
        "uncovered_candidates": total_uncovered,
        "total_annotations": total_annotations,
        "annotations_retained": total_annotations_retained,
        "annotations_missed": total_annotations_missed,
        "annotation_retention_percent": round(
            100 * total_annotations_retained / total_annotations, 2
        ) if total_annotations > 0 else 0.0,
    })

fieldnames = list(results[0].keys()) if results else []
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n=== SIZE EXPERIMENT RESULTS ===")
for row in results:
    print(
        f"Min area {row['min_component_area']} | "
        f"Candidates after filter: {row['candidate_hotspots_after_filter']} | "
        f"Covered: {row['covered_candidates']} | "
        f"Uncovered: {row['uncovered_candidates']} | "
        f"Annotations retained: {row['annotations_retained']}/{row['total_annotations']} "
        f"({row['annotation_retention_percent']}%)"
    )

print("\nCSV saved to:", CSV_PATH)