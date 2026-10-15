from pathlib import Path
import argparse

from ultralytics import YOLO

from prodml.pipeline import predict_image as run_pipeline


def predict_image(
    image_path: str | Path,
    model_path: str | Path,
    confidence: float = 0.25,
    save: bool = True,
):
    model = YOLO(str(model_path))
    image_path = Path(image_path)

    if image_path.exists():
        return run_pipeline(model, image_path, confidence, save=save)

    return model.predict(
        source=str(image_path),
        imgsz=416,
        conf=confidence,
        save=save,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument(
        "--model",
        default="model/weights/best.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    predict_image(args.image_path, args.model, args.confidence, save=True)


