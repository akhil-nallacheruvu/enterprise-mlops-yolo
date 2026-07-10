from ultralytics import YOLO
import time, numpy as np

def bench(model_path, label):
    model = YOLO(model_path)
    for _ in range(10):
        model("test_img.jpg", verbose=False)  # warmup
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model("test_img.jpg", verbose=False)
        times.append(time.perf_counter() - start)
    p50, p99 = np.percentile(times, [50, 99])
    print(f"{label:15s} p50={p50*1000:7.2f}ms  p99={p99*1000:7.2f}ms  throughput={1/np.mean(times):6.1f} img/s")

bench("yolov8n.pt", "PyTorch")
bench("yolov8n.onnx", "ONNX Runtime")
bench("yolov8n_openvino_model/", "OpenVINO")