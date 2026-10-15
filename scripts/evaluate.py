import argparse
from pathlib import Path
from ultralytics import YOLO

def evaluate_model(model_path: str | Path, data_yaml: str | Path, confidence: float = 0.25):
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(data = str (data_yaml), imgsz=416, conf=confidence)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/weights/best.pt")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        confidence=args.confidence,
    )

    print("Evaluation Metrics:")
    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)