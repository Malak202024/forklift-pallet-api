import sys
from pathlib import Path
import imagehash
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")
MAX_DIST = 4
EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def load(split):
    return {p.name: int(str(imagehash.phash(Image.open(p).convert("RGB"))), 16)
            for p in sorted((root / split / "images").glob("*")) if p.suffix.lower() in EXT}

h = {s: load(s) for s in ("train", "valid", "test")}

def leaks(a, b):
    hb = list(h[b].values())
    return [n for n, ha in h[a].items() if any(bin(ha ^ x).count("1") <= MAX_DIST for x in hb)]

for a, b in (("valid", "train"), ("test", "train"), ("test", "valid")):
    bad = leaks(a, b)
    print(f"{a} vs {b}: {len(bad)}/{len(h[a])} images ({100*len(bad)/max(len(h[a]),1):.1f}%) have a near-duplicate")