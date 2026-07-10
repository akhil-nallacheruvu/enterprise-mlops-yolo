# Production ML Serving Platform — Edge Vision Model

A production-oriented serving pipeline for a YOLOv8n object detection model, built to demonstrate end-to-end ML lifecycle ownership: optimization → packaging → containerized deployment → (upcoming) cloud deployment, CI/CD, and observability.

## Business Problem

Most ML models never leave the notebook they were trained in — or if they do, they run too slowly, too expensively, or with no visibility into whether they're still trustworthy once deployed. This project addresses the core production-readiness gap for CPU-bound edge deployment: taking a trained CV model and making it fast enough to serve in real time, portable enough to run anywhere, and instrumented enough to trust once it's live.

The detection model itself is derived from prior conservation-technology work (thermal/underwater wildlife monitoring), where edge devices are commonly CPU-only — making CPU inference optimization a directly relevant, real-world constraint rather than an artificial one.

## Week 1: Model Optimization & Containerized Serving

### Objective
Convert a baseline PyTorch model into an optimized, containerized inference service suitable for CPU-based edge deployment, with benchmarked evidence of the performance gain.

### Approach

1. **Baseline benchmark** — measured end-to-end inference latency (including preprocessing/postprocessing) for the unoptimized PyTorch model.
2. **Export & optimize** — converted the model to ONNX and to OpenVINO IR format (FP16), Intel's inference runtime purpose-built for CPU deployment.
3. **Apples-to-apples comparison** — benchmarked all three backends (PyTorch, ONNX Runtime, OpenVINO) through an identical harness: same image, same machine, same warmup protocol, 100 timed iterations each.
4. **API wrapper** — built a FastAPI service exposing `/predict` and `/health` endpoints, with the OpenVINO model as the serving backend.
5. **Containerization** — packaged the service in Docker for portable, reproducible deployment.

### Results

Benchmarked on an Intel 13th-gen i7-1355U (CPU-only, no GPU), 100 iterations per backend:

| Backend | p50 Latency | p99 Latency | Throughput |
|---|---|---|---|
| PyTorch (baseline) | 84.12 ms | 192.71 ms | 11.3 img/s |
| ONNX Runtime | 162.61 ms | 240.31 ms | 5.9 img/s |
| **OpenVINO (FP16)** | **29.00 ms** | **124.33 ms** | **29.2 img/s** |

**OpenVINO delivered a 65% reduction in p50 latency and a 2.6x throughput improvement over the PyTorch baseline.**

Model artifact size also dropped as part of the optimization: 12.3MB (ONNX) → 6.3MB (OpenVINO FP16).

### Why OpenVINO Won

OpenVINO is Intel's own inference runtime, built on the oneDNN backend with hand-tuned kernels for Intel instruction sets, and it compiles a hardware-aware execution graph (including scheduling across the CPU's Performance/Efficiency core split). Generic ONNX Runtime's default CPU execution provider doesn't get the same degree of hardware-specific tuning without additional manual configuration — which is consistent with the comparison numbers above.

**Note on generalizability:** this result is specific to Intel CPU hardware. On ARM (e.g. AWS Graviton) or GPU-backed infrastructure, this advantage would need to be re-benchmarked rather than assumed to transfer — a deliberate scope boundary, not an oversight.

### Serving Layer

The optimized OpenVINO model is served via a FastAPI application:

- `GET /health` — liveness check
- `POST /predict` — accepts an image file, returns detected objects (class ID, confidence, bounding box) and end-to-end request latency

Preprocessing implements YOLOv8n's expected letterbox resize (aspect-ratio-preserving, padded to 640x640) and normalization; postprocessing applies confidence thresholding and non-max suppression (NMS) to the raw `(1, 84, 8400)` output tensor, then rescales bounding boxes back to original image coordinates.

### Containerization

The service is packaged into a Docker image (`python:3.11-slim` base) with system dependencies for OpenCV (`libgl1`, `libglib2.0-0`) and verified to reproduce identical inference results to the local (non-containerized) run.

```bash
docker build -t enterprise-mlops-yolo .
docker run -d -p 8000:8000 --name yolo-service enterprise-mlops-yolo

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test_img.jpg"
```

Current image size: 1.28GB. This includes `ultralytics`/`torch`, which were only required for the export step, not for serving — a multi-stage build separating export-time and serving-time dependencies is a planned optimization to reduce this footprint.

### Stack
`Python` · `PyTorch` · `ONNX` · `OpenVINO` · `FastAPI` · `OpenCV` · `Docker`

---

## Roadmap

- [x] **Week 1** — Model optimization, benchmarking, FastAPI serving, containerization
- [ ] **Week 2** — Cloud deployment (AWS), MLflow model versioning, health checks/rollback
- [ ] **Week 3** — Airflow retraining pipeline, GitHub Actions CI/CD, dataset versioning
- [ ] **Week 4** — Prometheus/Grafana monitoring, load testing (Locust), drift detection
