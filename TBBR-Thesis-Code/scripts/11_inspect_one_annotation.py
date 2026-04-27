import json
import os
import cv2
import numpy as np

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

COCO_PATH = os.path.join(BASE_DIR, "raw_data", "test", "Flug1_105Media_coco.json")
IMAGE_DIR = os.path.join(BASE_DIR, "working", "yolo_dataset", "images", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "working", "inspect_multiple")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load COCO
with open(COCO_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# Limit number of images to inspect
N = 15

for idx, image_info in enumerate(images[:N]):
    image_id = image_info["id"]
    file_name = image_info["file_name"]

    parts = file_name.replace("\\", "/").split("/")
    block = parts[-2]
    stem = os.path.splitext(parts[-1])[0]

    img_name = f"{block}_{stem}.png"
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    anns = [a for a in annotations if a["image_id"] == image_id]

    for ann in anns:
        # Bounding box
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Polygon
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, f"inspect_{idx}.png")
    cv2.imwrite(out_path, img)

print("Saved multiple inspection images.")