import json
import os
import cv2
import numpy as np

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

TRAIN_JSON = os.path.join(BASE_DIR, "raw_data", "train", "Flug1_100-104Media_coco.json")
TEST_JSON = os.path.join(BASE_DIR, "raw_data", "test", "Flug1_105Media_coco.json")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "working", "yolo_dataset", "images", "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "working", "yolo_dataset", "images", "val")
TEST_IMG_DIR = os.path.join(BASE_DIR, "working", "yolo_dataset", "images", "test")

OUT_BASE = os.path.join(BASE_DIR, "working", "compare_splits")
OUT_TRAIN = os.path.join(OUT_BASE, "train")
OUT_VAL = os.path.join(OUT_BASE, "val")
OUT_TEST = os.path.join(OUT_BASE, "test")

for p in [OUT_TRAIN, OUT_VAL, OUT_TEST]:
    os.makedirs(p, exist_ok=True)


def build_unique_name(file_name):
    # example: images/Flug1_100/DJI_0004_R.npy
    parts = file_name.replace("\\", "/").split("/")
    block = parts[-2]
    stem = os.path.splitext(parts[-1])[0]
    return block, f"{block}_{stem}"


def draw_annotations_on_image(img_path, anns, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image:", img_path)
        return

    for ann in anns:
        # bbox in green
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # polygon in red
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    cv2.imwrite(out_path, img)


def sample_from_train_json():
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    train_count = 0
    val_count = 0

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        block, unique_name = build_unique_name(file_name)

        anns = ann_by_image.get(image_id, [])
        if len(anns) == 0:
            continue

        img_file = unique_name + ".png"

        # val block
        if block == "Flug1_104" and val_count < 3:
            img_path = os.path.join(VAL_IMG_DIR, img_file)
            out_path = os.path.join(OUT_VAL, f"{block}_{val_count}.png")
            if os.path.exists(img_path):
                draw_annotations_on_image(img_path, anns, out_path)
                val_count += 1

        # train blocks
        elif block in {"Flug1_100", "Flug1_101", "Flug1_102", "Flug1_103"} and train_count < 3:
            img_path = os.path.join(TRAIN_IMG_DIR, img_file)
            out_path = os.path.join(OUT_TRAIN, f"{block}_{train_count}.png")
            if os.path.exists(img_path):
                draw_annotations_on_image(img_path, anns, out_path)
                train_count += 1

        if train_count >= 3 and val_count >= 3:
            break


def sample_from_test_json():
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    test_count = 0

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        block, unique_name = build_unique_name(file_name)

        anns = ann_by_image.get(image_id, [])
        if len(anns) == 0:
            continue

        img_file = unique_name + ".png"
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        out_path = os.path.join(OUT_TEST, f"{block}_{test_count}.png")

        if os.path.exists(img_path):
            draw_annotations_on_image(img_path, anns, out_path)
            test_count += 1

        if test_count >= 3:
            break


if __name__ == "__main__":
    sample_from_train_json()
    sample_from_test_json()
    print("Saved comparison images in working/compare_splits/")