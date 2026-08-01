from fastapi import FastAPI, UploadFile, Response
from openvino import Core
import numpy as np
from PIL import Image
import io, time, cv2
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()
core = Core()
model = core.read_model("yolov8n_openvino_model/yolov8n.xml")
compiled_model = core.compile_model(model, "CPU")
output_layer = compiled_model.output(0)
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
BASELINE_MEAN_BRIGHTNESS = 114.77       #run compute_baseline_stats.py to get these numbers
BASELINE_STD_BRIGHTNESS = 1.0

REQUEST_COUNT = Counter(
    "predict_requests_total", "Total prediction requests", ["status"]
)

REQUEST_LATENCY = Histogram(
    "predict_latency_seconds", "Prediction request latency in seconds"
)

DRIFT_GAUGE = Gauge(
    "input_brightness_drift_zscore",
    "Z-score of current input brightness vs. reference baseline"
)

def preprocess(image: Image.Image):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    # letterbox resize (preserve aspect ratio, pad to square)
    scale = INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[0:new_h, 0:new_w] = resized

    # normalize, HWC -> CHW, add batch dim
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, (h, w)


def postprocess(result, scale, original_shape):
    # result shape: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = np.squeeze(result[0]).T  # (8400, 84)

    boxes = predictions[:, :4]        # cx, cy, w, h
    scores = predictions[:, 4:]       # 80 class scores
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > CONF_THRESHOLD
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return []

    # convert cx,cy,w,h -> x1,y1,x2,y2 (still in 640x640 letterboxed space)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)  # x,y,w,h for cv2.dnn.NMSBoxes

    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD
    )

    detections = []
    for i in np.array(indices).flatten():
        bx1, by1, bx2, by2 = x1[i] / scale, y1[i] / scale, x2[i] / scale, y2[i] / scale
        detections.append({
            "class_id": int(class_ids[i]),
            "confidence": float(confidences[i]),
            "box": [float(bx1), float(by1), float(bx2), float(by2)],
        })

    return detections

def compute_drift_score(image_array: np.ndarray) -> float:
    current_mean = image_array.mean()
    z_score = (current_mean - BASELINE_MEAN_BRIGHTNESS) / BASELINE_STD_BRIGHTNESS
    DRIFT_GAUGE.set(z_score)
    return z_score

@app.post("/predict")
async def predict(file: UploadFile):
    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image_array = np.array(image.convert("RGB"))
        drift_score = compute_drift_score(image_array)

        input_tensor, scale, original_shape = preprocess(image)
        result = compiled_model([input_tensor])[output_layer]
        detections = postprocess(result, scale, original_shape)
        latency_s = time.perf_counter() - start

        REQUEST_LATENCY.observe(latency_s)
        REQUEST_COUNT.labels(status="success").inc()

        return {
            "detections": detections,
            "latency_ms": latency_s * 1000,
            "drift_zscore": drift_score,
        }
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
