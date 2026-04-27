import os
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE = r"C:\Users\Lenovo\Desktop\TBBRdataset"

MODE = "thermal"   # "thermal" or "rgb"

INPUT_DIR = os.path.join(BASE, "raw_data")
OUTPUT_DIR = os.path.join(BASE, "working", "exported_png")

def process_split(split_name):
    input_path = os.path.join(INPUT_DIR, split_name, "images")
    output_path = os.path.join(OUTPUT_DIR, split_name)

    if not os.path.exists(input_path):
        print(f"{split_name} folder not found -> skipping")
        return

    os.makedirs(output_path, exist_ok=True)

    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing {split_name}/{folder}...")

        for file in tqdm(os.listdir(folder_path)):
            if not file.endswith(".npy"):
                continue

            npy_path = os.path.join(folder_path, file)

            try:
                arr = np.load(npy_path)

                if MODE == "thermal":
                    img = arr[:, :, 3]
                elif MODE == "rgb":
                    img = arr[:, :, [2, 1, 0]]
                else:
                    raise ValueError("MODE must be 'thermal' or 'rgb'")

                original_name = os.path.splitext(file)[0]
                unique_name = f"{folder}_{original_name}.png"
                save_path = os.path.join(output_path, unique_name)

                Image.fromarray(img).save(save_path)

            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    process_split("train")
    process_split("test")
    print("\nDone exporting images.")