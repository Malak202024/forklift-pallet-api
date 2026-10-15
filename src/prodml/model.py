from ultralytics import YOLO 
from pathlib import Path

def load_model(model_path: str | Path)-> YOLO:
     """loads a YOLO model"""
     model_path = Path(model_path)
     if not model_path.exists():
          raise ValueError(f"Model file not found: {model_path}")
     
     model = YOLO(str(model_path))
     return model
