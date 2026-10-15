from collections import Counter
from pathlib import Path

for split in ("train", "valid", "test"):
    c = Counter()
    for f in Path("dataset", split, "labels").glob("*.txt"):
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        if not rows:
            c["empty"] += 1
            continue
        classes = {int(float(r[0])) for r in rows}
        if classes == {0, 1}:
            c["both_classes"] += 1
        elif classes == {0}:
            c["forklift_only"] += 1
        else:
            c["pallet_only"] += 1
        kinds = {"polygon" if len(r) > 6 else "box" for r in rows}
        if len(kinds) == 2:
            c["mixed_box_and_polygon"] += 1
    print(split, dict(c))