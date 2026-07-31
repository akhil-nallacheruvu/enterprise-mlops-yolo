import mlflow

mlflow.set_experiment("yolo-edge-optimization")

# Run 1: PyTorch baseline
with mlflow.start_run(run_name="pytorch-baseline"):
    mlflow.log_param("backend", "PyTorch")
    mlflow.log_param("precision", "FP32")
    mlflow.log_metric("p50_latency_ms", 84.12)
    mlflow.log_metric("p99_latency_ms", 192.71)
    mlflow.log_metric("throughput_img_s", 11.3)

# Run 2: ONNX Runtime
with mlflow.start_run(run_name="onnxruntime"):
    mlflow.log_param("backend", "ONNX Runtime")
    mlflow.log_param("precision", "FP32")
    mlflow.log_metric("p50_latency_ms", 162.61)
    mlflow.log_metric("p99_latency_ms", 240.31)
    mlflow.log_metric("throughput_img_s", 5.9)
    mlflow.log_artifact("yolov8n.onnx")

# Run 3: OpenVINO (winner)
with mlflow.start_run(run_name="openvino-fp16"):
    mlflow.log_param("backend", "OpenVINO")
    mlflow.log_param("precision", "FP16")
    mlflow.log_metric("p50_latency_ms", 29.00)
    mlflow.log_metric("p99_latency_ms", 124.33)
    mlflow.log_metric("throughput_img_s", 29.2)
    mlflow.log_artifact("yolov8n_openvino_model/yolov8n.xml")
    mlflow.log_artifact("yolov8n_openvino_model/yolov8n.bin")
    