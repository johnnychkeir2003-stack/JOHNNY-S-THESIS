import json
import os
import cv2
import numpy as np
import csv

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

# -------------------------
# CHOOSE SPLIT TO ANALYZE
# -------------------------
SPLIT = "test"   # "train" or "test"

TRAIN_JSON = os.path.join(BASE_DIR, "raw_data", "train", "Flug1_100-104Media_coco.json")
TEST_JSON = os.path.join(BASE_DIR, "raw_data", "test", "Flug1_105Media_coco.json")

TRAIN_RAW = os.path.join(BASE_DIR, "raw_data", "train", "images")
TEST_RAW = os.path.join(BASE_DIR, "raw_data", "test", "images")

OUTPUT_DIR = os.path.join(BASE_DIR, "working", "uncovered_candidates_filtered")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, f"{SPLIT}_candidate_counts_filtered.csv")

# -------------------------
# PARAMETERS
# -------------------------
THERMAL_PERCENTILE = 98     # top hot pixels
HEIGHT_PERCENTILE = 60      # rough building mask from height channel
MIN_COMPONENT_AREA = 50     # remove tiny noise
MIN_ELONGATION = 3.0        # keep long/thin structures only
SAVE_EXAMPLES = 10          # save first N overlay examples

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
    # example: Flug1_105/DJI_0004_R.npy
    parts = file_name.replace("\\", "/").split("/")
    block = parts[-2]
    fname = parts[-1]
    return os.path.join(raw_base, block, fname), block, os.path.splitext(fname)[0]


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
    # rough building-like mask using height
    h_thr = np.percentile(height_channel, HEIGHT_PERCENTILE)
    building_mask = (height_channel >= h_thr)

    # thermal threshold only inside building-like areas
    if np.any(building_mask):
        t_thr = np.percentile(thermal[building_mask], THERMAL_PERCENTILE)
    else:
        t_thr = np.percentile(thermal, THERMAL_PERCENTILE)

    hot_mask = (thermal >= t_thr)

    candidate_mask = (building_mask & hot_mask).astype(np.uint8) * 255

    # light cleanup
    kernel = np.ones((3, 3), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    return candidate_mask, t_thr, h_thr


def component_masks(binary_mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    comps = []
    for label_id in range(1, n_labels):  # 0 = background
        area = stats[label_id, cv2.CC_STAT_AREA]
        comp = (labels == label_id).astype(np.uint8) * 255
        comps.append((label_id, comp, area))

    return comps


def filter_components(comps):
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

        # filters
        if area < MIN_COMPONENT_AREA:
            continue

        if elongation < MIN_ELONGATION:
            continue

        filtered.append({
            "mask": comp_mask,
            "area": int(area),
            "elongation": float(elongation),
            "bbox": (int(x1), int(y1), int(width), int(height)),
        })

    return filtered


def component_overlaps_annotation(comp_mask, ann_mask):
    overlap = cv2.bitwise_and(comp_mask, ann_mask)
    return np.count_nonzero(overlap) > 0


def save_overlay(thermal, ann_mask, filtered_comps, covered_flags, out_path):
    vis = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)

    # annotation polygons in red
    vis[ann_mask > 0] = (0, 0, 255)

    # candidates:
    # green = covered
    # yellow = uncovered
    for comp, is_covered in zip(filtered_comps, covered_flags):
        x1, y1, width, height = comp["bbox"]
        color = (0, 255, 0) if is_covered else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x1 + width, y1 + height), color, 2)

    cv2.imwrite(out_path, vis)


# -------------------------
# MAIN
# -------------------------
json_path, raw_base = load_config()

with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

ann_by_image = {}
for ann in annotations:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

total_candidates_before_filter = 0
total_candidates_after_filter = 0
total_covered = 0
total_uncovered = 0
saved_examples = 0

rows = []

for img_info in images:
    image_id = img_info["id"]
    file_name = img_info["file_name"]
    img_height = img_info["height"]
    img_width = img_info["width"]

    raw_path, block, stem = file_name_to_raw_path(raw_base, file_name)

    if not os.path.exists(raw_path):
        print("Missing raw file:", raw_path)
        continue

    arr = np.load(raw_path)
    thermal = arr[:, :, 3]
    height_channel = arr[:, :, 4]

    anns = ann_by_image.get(image_id, [])
    ann_mask = build_annotation_mask(img_height, img_width, anns)

    candidate_mask, t_thr, h_thr = build_candidate_mask(thermal, height_channel)
    comps = component_masks(candidate_mask)
    filtered_comps = filter_components(comps)

    total_candidates_before_filter += len(comps)
    total_candidates_after_filter += len(filtered_comps)

    covered = 0
    uncovered = 0
    covered_flags = []

    for comp in filtered_comps:
        is_covered = component_overlaps_annotation(comp["mask"], ann_mask)
        covered_flags.append(is_covered)

        if is_covered:
            covered += 1
        else:
            uncovered += 1

    total_covered += covered
    total_uncovered += uncovered

    mean_area = np.mean([c["area"] for c in filtered_comps]) if filtered_comps else 0
    mean_elongation = np.mean([c["elongation"] for c in filtered_comps]) if filtered_comps else 0

    rows.append({
        "image_id": image_id,
        "file_name": file_name,
        "num_annotations": len(anns),
        "candidates_before_filter": len(comps),
        "candidates_after_filter": len(filtered_comps),
        "covered_candidates": covered,
        "uncovered_candidates": uncovered,
        "thermal_threshold": round(float(t_thr), 2),
        "height_threshold": round(float(h_thr), 2),
        "mean_candidate_area": round(float(mean_area), 2),
        "mean_candidate_elongation": round(float(mean_elongation), 2),
    })

    if saved_examples < SAVE_EXAMPLES:
        out_img = os.path.join(OUTPUT_DIR, f"{block}_{stem}_overlay.png")
        save_overlay(thermal, ann_mask, filtered_comps, covered_flags, out_img)
        saved_examples += 1

fieldnames = [
    "image_id",
    "file_name",
    "num_annotations",
    "candidates_before_filter",
    "candidates_after_filter",
    "covered_candidates",
    "uncovered_candidates",
    "thermal_threshold",
    "height_threshold",
    "mean_candidate_area",
    "mean_candidate_elongation",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n=== SUMMARY ===")
print("Split:", SPLIT)
print("Candidate hotspots before filtering:", total_candidates_before_filter)
print("Candidate hotspots after filtering:", total_candidates_after_filter)
print("Covered by annotations:", total_covered)
print("Not covered by annotations:", total_uncovered)
print("CSV saved to:", CSV_PATH)
print("Overlay examples saved in:", OUTPUT_DIR)