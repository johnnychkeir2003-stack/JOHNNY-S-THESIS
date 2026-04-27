import json
import os
import cv2
import numpy as np

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

# -------------------------
# CONFIG
# -------------------------
SPLIT = "test"  # "train" or "test"
MAX_SAMPLES = 15

TRAIN_JSON = os.path.join(BASE_DIR, "raw_data", "train", "Flug1_100-104Media_coco.json")
TEST_JSON = os.path.join(BASE_DIR, "raw_data", "test", "Flug1_105Media_coco.json")

TRAIN_RAW = os.path.join(BASE_DIR, "raw_data", "train", "images")
TEST_RAW = os.path.join(BASE_DIR, "raw_data", "test", "images")

OUT_DIR = os.path.join(BASE_DIR, "working", "manual_subset")
os.makedirs(OUT_DIR, exist_ok=True)

# filtering parameters (LESS strict now)
THERMAL_PERCENTILE = 98
HEIGHT_PERCENTILE = 60
MIN_AREA = 30
MIN_ELONGATION = 2.0

# -------------------------
# HELPERS
# -------------------------
def load_config():
    if SPLIT == "train":
        return TRAIN_JSON, TRAIN_RAW
    return TEST_JSON, TEST_RAW


def file_to_path(raw_base, file_name):
    parts = file_name.replace("\\", "/").split("/")
    block = parts[-2]
    fname = parts[-1]
    return os.path.join(raw_base, block, fname), block, os.path.splitext(fname)[0]


def build_ann_mask(h, w, anns):
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
    return mask


def get_candidates(thermal, height):
    h_thr = np.percentile(height, HEIGHT_PERCENTILE)
    building = height >= h_thr

    t_thr = np.percentile(thermal[building], THERMAL_PERCENTILE) if np.any(building) else np.percentile(thermal, THERMAL_PERCENTILE)
    hot = thermal >= t_thr

    mask = (building & hot).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_components(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = (labels == i).astype(np.uint8) * 255
        comps.append((comp, area))
    return comps


def filter_components(comps):
    filtered = []
    for comp, area in comps:
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        w = x2 - x1 + 1
        h = y2 - y1 + 1

        if min(w, h) == 0:
            continue

        elong = max(w, h) / min(w, h)

        if area < MIN_AREA:
            continue
        if elong < MIN_ELONGATION:
            continue

        filtered.append((comp, (x1, y1, w, h)))

    return filtered


def overlaps(comp, ann_mask):
    return np.count_nonzero(cv2.bitwise_and(comp, ann_mask)) > 0


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

saved = 0

for img_info in images:
    if saved >= MAX_SAMPLES:
        break

    img_id = img_info["id"]
    file_name = img_info["file_name"]
    h = img_info["height"]
    w = img_info["width"]

    raw_path, block, stem = file_to_path(raw_base, file_name)
    if not os.path.exists(raw_path):
        continue

    arr = np.load(raw_path)
    thermal = arr[:, :, 3]
    height = arr[:, :, 4]

    anns = ann_by_image.get(img_id, [])
    ann_mask = build_ann_mask(h, w, anns)

    mask = get_candidates(thermal, height)
    comps = extract_components(mask)
    comps = filter_components(comps)

    for comp, (x1, y1, ww, hh) in comps:
        if overlaps(comp, ann_mask):
            continue  # we only want uncovered

        # SAVE FULL IMAGE WITH BOX
        vis = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
        vis[ann_mask > 0] = (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x1 + ww, y1 + hh), (0, 255, 255), 2)

        full_path = os.path.join(OUT_DIR, f"{saved}_full.png")
        crop_path = os.path.join(OUT_DIR, f"{saved}_crop.png")

        cv2.imwrite(full_path, vis)

        # SAVE CROPPED REGION
        crop = vis[y1:y1 + hh, x1:x1 + ww]
        if crop.size > 0:
            cv2.imwrite(crop_path, crop)

        saved += 1

        if saved >= MAX_SAMPLES:
            break

print(f"Saved {saved} uncovered samples for manual review in:", OUT_DIR)