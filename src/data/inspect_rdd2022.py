from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


DATASET_DIR = Path("data/raw/RDD2022")


def inspect_dataset(dataset_dir: Path):
    image_extensions = [".jpg", ".jpeg", ".png"]

    images = [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    xml_files = list(dataset_dir.rglob("*.xml"))

    print("=" * 60)
    print("RDD2022 DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset path: {dataset_dir}")
    print(f"Images found: {len(images)}")
    print(f"XML annotations found: {len(xml_files)}")

    class_counter = Counter()

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_counter[class_name] += 1

        except Exception as error:
            print(f"Could not read XML: {xml_path}")
            print(error)

    print("\nDamage classes found:")
    for class_name, count in class_counter.most_common():
        print(f"{class_name}: {count}")

    print("\nExample images:")
    for image_path in images[:5]:
        print(image_path)

    print("\nExample XML files:")
    for xml_path in xml_files[:5]:
        print(xml_path)


if __name__ == "__main__":
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}. "
            "Please extract RDD2022 into data/raw/RDD2022 first."
        )

    inspect_dataset(DATASET_DIR)