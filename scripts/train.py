import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(
    model_name: str,
    data_yaml: str,
    epochs: int = 40,
    imgsz: int = 416,
    name: str = "forklift-pallet-2",
):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=0,
        project="runs/train",
        name=name,
        exist_ok=True,
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--name", default="forklift-pallet-2")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=args.name,
    )