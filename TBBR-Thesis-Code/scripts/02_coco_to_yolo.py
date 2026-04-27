import os
import json

BASE = r"C:\Users\Lenovo\Desktop\TBBRdataset"

TEST_JSON = os.path.join(BASE, "raw_data", "test", "Flug1_105Media_coco.json")
TRAIN_JSON = os.path.join(BASE, "raw_data", "train", "Flug1_100-104Media_coco.json")

OUTPUT_BASE = os.path.join(BASE, "working", "yolo_labels")


def coco_bbox_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return x_center, y_center, w_norm, h_norm


def build_unique_name(file_name):
    # example input from JSON:
    # images/Flug1_100/DJI_0004_R.npy
    # output:
    # Flug1_100_DJI_0004_R
    parts = file_name.replace("\\", "/").split("/")
    folder = parts[-2]
    stem = os.path.splitext(parts[-1])[0]
    return f"{folder}_{stem}"


def process_coco(json_path, split_name):
    if not os.path.exists(json_path):
        print(f"{split_name} annotation file not found -> skipping")
        return

    output_dir = os.path.join(OUTPUT_BASE, split_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_info = {}
    for img in images:
        unique_name = build_unique_name(img["file_name"])
        image_info[img["id"]] = {
            "file_name": unique_name,
            "width": img["width"],
            "height": img["height"],
        }

    labels_per_image = {}

    for ann in annotations:
        image_id = ann["image_id"]

        if image_id not in image_info:
            continue

        bbox = ann.get("bbox")
        if bbox is None:
            continue

        width = image_info[image_id]["width"]
        height = image_info[image_id]["height"]
        file_stem = image_info[image_id]["file_name"]

        x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(bbox, width, height)

        line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        if file_stem not in labels_per_image:
            labels_per_image[file_stem] = []

        labels_per_image[file_stem].append(line)

    for file_stem, lines in labels_per_image.items():
        txt_path = os.path.join(output_dir, file_stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"{split_name}: created {len(labels_per_image)} label files in {output_dir}")


if __name__ == "__main__":
    process_coco(TRAIN_JSON, "train")
    process_coco(TEST_JSON, "test")
    print("Done converting COCO to YOLO.")