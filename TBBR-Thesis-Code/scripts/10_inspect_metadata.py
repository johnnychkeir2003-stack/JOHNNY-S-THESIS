import json
import os

BASE = r"C:\Users\Lenovo\Desktop\TBBRdataset"
META_DIR = os.path.join(BASE, "metadata")

FILES_TO_CHECK = [
    "Flug1_100_stac_spec.json",
    "Flug1_105_stac_spec.json",
    "Flug1_collection_stac_spec.json",
    "Flug1_100-105_frictionless_standards.json"
]

def print_dict_keys(title, obj):
    print(f"\n{title}")
    if isinstance(obj, dict):
        for key in obj.keys():
            print(" -", key)
    else:
        print(obj)

for file_name in FILES_TO_CHECK:
    file_path = os.path.join(META_DIR, file_name)

    print("\n" + "=" * 60)
    print("Inspecting:", file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_dict_keys("Top-level keys:", data)

    if "bbox" in data:
        print("\nTop-level bbox:")
        print(data["bbox"])

    if "geometry" in data:
        print("\nTop-level geometry:")
        print(data["geometry"])

    if "properties" in data:
        print_dict_keys("Top-level properties keys:", data["properties"])

    if "links" in data:
        print("\nFirst 5 links:")
        links = data["links"]
        for link in links[:5]:
            print(link)

    if "assets" in data:
        print_dict_keys("Top-level assets keys:", data["assets"])

    if "resources" in data:
        print("\nNumber of resources:", len(data["resources"]))
        first = data["resources"][0]
        print("\nFirst resource:")
        print(first)