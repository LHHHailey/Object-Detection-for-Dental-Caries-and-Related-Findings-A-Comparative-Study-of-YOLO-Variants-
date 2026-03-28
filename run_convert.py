import json
import os
import shutil
from pathlib import Path

SRC_DIR = Path("dataset")
DEST_DIR = Path("yolo_dataset")


def ensure_dest():
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    for split in ("train", "val", "test"):
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def write_yaml(class_names):
    with open(DEST_DIR / "data.yaml", "w", encoding="utf-8") as fp:
        fp.write("train: ../train/images\n")
        fp.write("val: ../val/images\n")
        fp.write("test: ../test/images\n")
        fp.write(f"nc: {len(class_names)}\n")
        fp.write(f"names: {class_names}\n")


def xyxy_to_yolo(left, top, width, height, img_width, img_height):
    x_center = (left + width / 2) / img_width
    y_center = (top + height / 2) / img_height
    w = width / img_width
    h = height / img_height
    return x_center, y_center, w, h


def convert_split(src_split: str, dst_split: str, class_to_idx: dict):
    shutil.copytree(
        SRC_DIR / src_split / "img",
        DEST_DIR / dst_split / "images",
        dirs_exist_ok=True,
    )

    ann_dir = SRC_DIR / src_split / "ann"
    for ann_file in ann_dir.glob("*.json"):
        ann = json.loads(ann_file.read_text(encoding="utf-8"))
        img_width = ann["size"]["width"]
        img_height = ann["size"]["height"]
        out_name = ann_file.name.replace(".jpg.json", ".txt")
        out_path = DEST_DIR / dst_split / "labels" / out_name

        lines = []
        for obj in ann.get("objects", []):
            cls = obj.get("classTitle")
            if cls not in class_to_idx:
                continue
            pts = obj.get("points", {}).get("exterior", [])
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            left, top = min(xs), min(ys)
            width = max(xs) - left
            height = max(ys) - top
            x, y, w, h = xyxy_to_yolo(left, top, width, height, img_width, img_height)
            lines.append(f"{class_to_idx[cls]} {x} {y} {w} {h}")

        out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    meta = json.loads((SRC_DIR / "meta.json").read_text(encoding="utf-8"))
    class_names = [c["title"] for c in meta["classes"]]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    ensure_dest()
    write_yaml(class_names)

    dirs_map = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in dirs_map.items():
        convert_split(src_split, dst_split, class_to_idx)

    print("Converted dataset saved to:", DEST_DIR.resolve())
    for split in ("train", "val", "test"):
        n_img = len(list((DEST_DIR / split / "images").glob("*.jpg")))
        n_lbl = len(list((DEST_DIR / split / "labels").glob("*.txt")))
        print(f"{split}: images={n_img}, labels={n_lbl}")


if __name__ == "__main__":
    main()
