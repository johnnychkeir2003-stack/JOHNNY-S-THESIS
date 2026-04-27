import os
import json
import shutil

BASE = r"C:\Users\Lenovo\Desktop\TBBRdataset"

TRAIN_JSON = os.path.join(BASE, "raw_data", "train", "Flug1_100-104Media_coco.json")
TEST_JSON = os.path.join(BASE, "raw_data", "test", "Flug1_105Media_coco.json")

PNG_DIR = os.path.join(BASE, "working", "exported_png")
LABEL_DIR = os.path.join(BASE, "working", "yolo_labels")
OUT_DIR = os.path.join(BASE, "working", "yolo_dataset")

TRAIN_BLOCKS = {"Flug1_100", "Flug1_101", "Flug1_102", "Flug1_103"}
VAL_BLOCKS = {"Flug1_104"}
TEST_BLOCKS = {"Flug1_105"}


def ensure_dirs():
    for part in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(OUT_DIR, part, split), exist_ok=True)


def clear_output_dirs():
    for part in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            folder = os.path.join(OUT_DIR, part, split)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)


def build_unique_name(file_name):
    # input example: images/Flug1_100/DJI_0004_R.npy
    # output: Flug1_100_DJI_0004_R
    parts = file_name.replace("\\", "/").split("/")
    folder = parts[-2]
    stem = os.path.splitext(parts[-1])[0]
    return folder, f"{folder}_{stem}"


def collect_names_from_coco(json_path, allowed_blocks):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    names = set()

    for img in coco.get("images", []):
        block, unique_name = build_unique_name(img["file_name"])
        if block in allowed_blocks:
            names.add(unique_name)

    return names


def copy_split(image_names, source_split, target_split):
    src_img_dir = os.path.join(PNG_DIR, source_split)
    src_lbl_dir = os.path.join(LABEL_DIR, source_split)

    dst_img_dir = os.path.join(OUT_DIR, "images", target_split)
    dst_lbl_dir = os.path.join(OUT_DIR, "labels", target_split)

    copied_images = 0
    copied_labels = 0
    missing_labels = 0
    missing_images = 0

    for name in sorted(image_names):
        img_file = name + ".png"
        lbl_file = name + ".txt"

        src_img = os.path.join(src_img_dir, img_file)
        src_lbl = os.path.join(src_lbl_dir, lbl_file)

        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(dst_img_dir, img_file))
            copied_images += 1
        else:
            missing_images += 1

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, lbl_file))
            copied_labels += 1
        else:
            # normal: some images have no annotations
            missing_labels += 1

    print(f"\n{target_split.upper()} SUMMARY")
    print(f"Images copied: {copied_images}")
    print(f"Labels copied: {copied_labels}")
    print(f"Images without labels: {missing_labels}")
    print(f"Missing image files: {missing_images}")


def main():
    ensure_dirs()
    clear_output_dirs()

    train_names = collect_names_from_coco(TRAIN_JSON, TRAIN_BLOCKS)
    val_names = collect_names_from_coco(TRAIN_JSON, VAL_BLOCKS)
    test_names = collect_names_from_coco(TEST_JSON, TEST_BLOCKS)

    print(f"Train image names found in COCO: {len(train_names)}")
    print(f"Val image names found in COCO: {len(val_names)}")
    print(f"Test image names found in COCO: {len(test_names)}")

    copy_split(train_names, "train", "train")
    copy_split(val_names, "train", "val")
    copy_split(test_names, "test", "test")

    print("\nDone building block-based YOLO dataset from COCO-defined images.")


if __name__ == "__main__":
    main()