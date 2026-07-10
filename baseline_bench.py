# baseline_bench.py
from ultralytics import YOLO
import time
import numpy as np
import onnxruntime as ort

model = YOLO("yolov8n.pt")  # swap in your trained weights
dummy_input = "test_img.jpg"

# warmup
for _ in range(10):
    model(dummy_input, verbose=False)

# benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    model(dummy_input, verbose=False)
    times.append(time.perf_counter() - start)

print(f"PyTorch baseline: p50={np.percentile(times,50)*1000:.2f}ms, "
      f"p99={np.percentile(times,99)*1000:.2f}ms")

model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12, simplify=True)
model.export(format="openvino", half=True)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("yolov8n.onnx", sess_options, providers=["CPUExecutionProvider"])