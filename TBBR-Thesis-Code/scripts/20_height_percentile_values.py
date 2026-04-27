import os
import numpy as np

BASE_DIR = r"C:\Users\Lenovo\Desktop\TBBRdataset"

SPLIT = "test"
MAX_IMAGES = 30

TRAIN_RAW = os.path.join(BASE_DIR, "raw_data", "train", "images")
TEST_RAW = os.path.join(BASE_DIR, "raw_data", "test", "images")

HEIGHT_PERCENTILES = [40, 50, 60, 70]

def load_raw_base():
    if SPLIT == "train":
        return TRAIN_RAW
    else:
        return TEST_RAW

raw_base = load_raw_base()

all_files = []
for root, _, files in os.walk(raw_base):
    for f in files:
        if f.endswith(".npy"):
            all_files.append(os.path.join(root, f))

all_files = all_files[:MAX_IMAGES]

print("\n=== HEIGHT PERCENTILE VALUES (IN METERS) ===\n")

results = {p: [] for p in HEIGHT_PERCENTILES}

for path in all_files:
    arr = np.load(path)
    height_channel = arr[:, :, 4]  # same as your pipeline

    valid = height_channel[np.isfinite(height_channel)]

    for p in HEIGHT_PERCENTILES:
        val = np.percentile(valid, p)
        results[p].append(val)

# Print summary
for p in HEIGHT_PERCENTILES:
    vals = np.array(results[p])
    print(f"Percentile {p}%:")
    print(f"  Mean: {np.mean(vals):.2f} m")
    print(f"  Min : {np.min(vals):.2f} m")
    print(f"  Max : {np.max(vals):.2f} m")
    print()