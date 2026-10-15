from pathlib import Path
from ultralytics import YOLO

def predict_image(model: YOLO, image_path:str | Path, confidence: float =0.25, save: bool = False)->list:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    return model.predict(
            source=str(image_path),
            imgsz=416,
            conf = confidence,
            save=save,
        )
