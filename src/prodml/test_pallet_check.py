from pathlib import Path
import cv2
from ultralytics import YOLO

m = YOLO("weights/best.pt")
out = Path("outputs/test_pallet_check")
out.mkdir(parents=True, exist_ok=True)
imgs = sorted(Path("dataset/test/images").glob("*"))
hits = 0
for p in imgs:
    r = m.predict(str(p), imgsz=416, conf=0.25, verbose=False)[0]
    if any(m.names[int(c)] == "pallet" for c in r.boxes.cls.tolist()):
        hits += 1
        if hits <= 20:
            cv2.imwrite(str(out / p.name), r.plot())
print(f"{hits}/{len(imgs)} forklift-only test images have a pallet predicted at conf >= 0.25")