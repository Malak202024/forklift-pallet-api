import io

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from prodml.api.schema import DetectionItem, PredictResponse
from prodml.model import load_model

app = FastAPI(title="Forklift/Pallet Detection API")

MODEL = load_model("model/weights/best.pt")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = MODEL(image, conf=0.25, imgsz=416)

    detections: list[DetectionItem] = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = result.names[cls_id]

            box_xyxy = box.xyxy[0]
            if hasattr(box_xyxy, "tolist"):
                x1, y1, x2, y2 = box_xyxy.tolist()
            else:
                x1, y1, x2, y2 = box_xyxy

            detections.append(
                DetectionItem(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    x1=round(float(x1), 2),
                    y1=round(float(y1), 2),
                    x2=round(float(x2), 2),
                    y2=round(float(y2), 2),
                )
            )

    return PredictResponse(status="ok", detections=detections)