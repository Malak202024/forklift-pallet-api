import shutil
import sys
from pathlib import Path

import cv2
import imagehash
import yaml
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
out = Path("dataset_clean")
MAX_DIST = 4
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load(split):
    return {p: int(str(imagehash.phash(Image.open(p).convert("RGB"))), 16)
            for p in sorted((root / split / "images").glob("*"))
            if p.suffix.lower() in EXT}


h = {s: load(s) for s in ("train", "valid", "test")}
train_items = list(h["train"].items())


def nearest(ha):
    path, hb = min(train_items, key=lambda kv: bin(ha ^ kv[1]).count("1"))
    return path, bin(ha ^ hb).count("1")


if out.exists():
    shutil.rmtree(out)
pairs = Path("outputs/dup_pairs")
pairs.mkdir(parents=True, exist_ok=True)

for split in ("valid", "test"):
    kept = dup = 0
    (out / split / "images").mkdir(parents=True)
    (out / split / "labels").mkdir(parents=True)
    for p, ha in h[split].items():
        match, dist = nearest(ha)
        if dist <= MAX_DIST:
            dup += 1
            if dup <= 8:  # left = valid/test image, right = nearest train image
                a, b = cv2.imread(str(p)), cv2.imread(str(match))
                if a is not None and b is not None:
                    a, b = (cv2.resize(x, (320, 320)) for x in (a, b))
                    cv2.imwrite(str(pairs / f"{split}_{dup}_d{dist}.jpg"), cv2.hconcat([a, b]))
        else:
            kept += 1
            shutil.copy2(p, out / split / "images" / p.name)
            lab = root / split / "labels" / (p.stem + ".txt")
            if lab.exists():
                shutil.copy2(lab, out / split / "labels" / lab.name)
    print(f"{split}: kept {kept} clean, dropped {dup} near-duplicates")

(out / "data.yaml").write_text(yaml.safe_dump({
    "path": str(out.resolve()),
    "train": "valid/images", "val": "valid/images", "test": "test/images",
    "names": {0: "forklift", 1: "pallet"},
}))