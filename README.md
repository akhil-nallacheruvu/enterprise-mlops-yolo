# Production ML Serving Platform — Edge Vision Model

An end-to-end MLOps system that takes a computer vision model from a trained artifact to a monitored, load-tested, publicly deployed production service — covering optimization, serving, containerization, cloud deployment, experiment tracking, dataset/model versioning, pipeline orchestration, CI/CD, and observability.

## Results at a Glance

| Metric | Result |
|---|---|
| Inference latency (p50, single request) | **29ms** (65% reduction vs. unoptimized baseline) |
| Throughput (single request) | **29.2 img/s** (2.6x improvement vs. baseline) |
| Sustained throughput (20 concurrent users) | **7.51 req/s, 0 failures** over a 3-minute load test |
| Estimated cost per inference | **~$0.0000008** (≈ $0.77 per 1M inferences, `t3.small` on-demand rate) |
| Deployment | Live on AWS EC2, auto-recovers from crash/reboot, deployed via CI/CD on every merge |
| Drift detection | Validated: reference input → z-score ≈ 0; out-of-distribution input → z-score ≈ 3.9 |

## Business Problem

Most ML models never leave the notebook they were trained in — or if they do, they run too slowly, too expensively, or with no visibility into whether they're still trustworthy once deployed. This project addresses the core production-readiness gap for CPU-bound edge deployment: taking a trained CV model and making it fast enough to serve in real time, portable enough to run anywhere, versioned and orchestrated like a real system, load-tested under realistic traffic, and instrumented enough to trust once it's live.

The detection model itself is derived from prior conservation-technology work (thermal/underwater wildlife monitoring), where edge devices are commonly CPU-only — making CPU inference optimization a directly relevant, real-world constraint rather than an artificial one.

---

## System Overview

| Layer | What it does | Tooling |
|---|---|---|
| **Optimization** | Converts a trained model into a fast, CPU-tuned inference artifact | PyTorch, ONNX, OpenVINO |
| **Serving** | Exposes the optimized model over HTTP | FastAPI, OpenCV |
| **Packaging** | Makes the service portable and reproducible | Docker |
| **Deployment** | Runs the service on a publicly reachable host | AWS EC2 |
| **Experiment tracking** | Records and compares optimization runs | MLflow |
| **Data/model versioning** | Tracks large binary artifacts outside of git | DVC + S3 |
| **Orchestration** | Models the retraining lifecycle as a pipeline | Apache Airflow |
| **CI/CD** | Tests, builds, and deploys automatically on merge | GitHub Actions, GHCR |
| **Observability** | Metrics, dashboards, load testing, drift detection | Prometheus, Grafana, Locust |

---

## Getting Started

### Prerequisites
- Docker
- Python 3.11
- AWS account (for S3/DVC and EC2 deployment)
- `dvc[s3]` installed locally (`pip install dvc[s3]`)

### Run the service locally

```bash
git clone <repo-url>
cd <repo-directory>

# pull versioned model + data artifacts from the S3-backed DVC remote
dvc pull

docker build -t enterprise-mlops-yolo .
docker run -d -p 8000:8000 --name yolo-service enterprise-mlops-yolo

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/test_img.jpg"
```

**Example response:**

```json
{
  "detections": [
    {"class_id": 16, "confidence": 0.906, "box": [174.1, 299.9, 410.6, 555.5]}
  ],
  "latency_ms": 29.4,
  "drift_zscore": 0.0008
}
```

### Run the monitoring stack locally

```bash
cd monitoring
docker compose up -d
```

- Prometheus: `http://localhost:9090` (check **Status → Targets** for the service's health)
- Grafana: `http://localhost:3000` (login `admin` / `admin`) — add Prometheus as a data source (`http://prometheus:9090`) and build dashboards for request volume, p50/p99 latency, error rate, and drift z-score

### Run a load test

```bash
pip install locust
locust -f locustfile.py --host http://<deployment-host>:8000 \
  --users 20 --spawn-rate 5 --run-time 3m --headless
```

### Run the orchestration pipeline locally

```bash
cd airflow
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```

Open `http://localhost:8080` (default login `airflow` / `airflow`) — the `yolo_retrain_pipeline` DAG is listed and runnable from the UI.

### Deploy your own copy

See [Cloud Deployment](#cloud-deployment-aws-ec2) and [CI/CD Pipeline](#cicd-pipeline-github-actions) for full setup, including the repository secrets required for automated deployment.

---

## Model Optimization & Containerized Serving

### Objective
Convert a baseline PyTorch model into an optimized, containerized inference service suitable for CPU-based edge deployment, with benchmarked evidence of the performance gain.

### Approach

1. **Baseline benchmark** — measured end-to-end inference latency (including preprocessing/postprocessing) for the unoptimized PyTorch model.
2. **Export & optimize** — converted the model to ONNX and to OpenVINO IR format (FP16), Intel's inference runtime purpose-built for CPU deployment.
3. **Apples-to-apples comparison** — benchmarked all three backends (PyTorch, ONNX Runtime, OpenVINO) through an identical harness: same image, same machine, same warmup protocol, 100 timed iterations each.
4. **API wrapper** — built a FastAPI service exposing `/predict`, `/health`, and `/metrics` endpoints, with the OpenVINO model as the serving backend.
5. **Containerization** — packaged the service in Docker for portable, reproducible deployment.

### Results

Benchmarked on an Intel 13th-gen i7 (CPU-only, no GPU), 100 iterations per backend:

| Backend | p50 Latency | p99 Latency | Throughput |
|---|---|---|---|
| PyTorch (baseline) | 84.12 ms | 192.71 ms | 11.3 img/s |
| ONNX Runtime | 162.61 ms | 240.31 ms | 5.9 img/s |
| **OpenVINO (FP16)** | **29.00 ms** | **124.33 ms** | **29.2 img/s** |

**OpenVINO delivered a 65% reduction in p50 latency and a 2.6x throughput improvement over the PyTorch baseline.** Model artifact size also dropped: 12.3MB (ONNX) → 6.3MB (OpenVINO FP16).

**Why OpenVINO won:** it's Intel's own inference runtime, built on the oneDNN backend with hand-tuned kernels for Intel instruction sets, and it compiles a hardware-aware execution graph (including scheduling across the CPU's Performance/Efficiency core split). Generic ONNX Runtime's default CPU execution provider doesn't get the same degree of hardware-specific tuning without additional manual configuration.

**Note on generalizability:** this result is specific to Intel CPU hardware. On ARM (e.g. AWS Graviton) or GPU-backed infrastructure, this advantage would need to be re-benchmarked rather than assumed to transfer.

### Serving Layer

- `GET /health` — liveness check
- `GET /metrics` — Prometheus-format metrics (request counts, latency histogram, drift gauge)
- `POST /predict` — accepts an image file, returns detected objects (class ID, confidence, bounding box), end-to-end request latency, and an input drift z-score

Preprocessing implements YOLOv8n's expected letterbox resize (aspect-ratio-preserving, padded to 640x640) and normalization; postprocessing applies confidence thresholding and non-max suppression (NMS) to the raw `(1, 84, 8400)` output tensor, then rescales bounding boxes back to original image coordinates.

### Containerization

Packaged into a Docker image (`python:3.11-slim` base) with system dependencies for OpenCV (`libgl1`, `libglib2.0-0`), verified to reproduce identical inference results to the local (non-containerized) run. Current image size: 1.28GB — includes `ultralytics`/`torch`, only required at export time, not serving time; separating these into a multi-stage build is a flagged future optimization.

### Stack
`Python` · `PyTorch` · `ONNX` · `OpenVINO` · `FastAPI` · `OpenCV` · `Docker`

---

## Cloud Deployment (AWS EC2)

### Setup steps

```bash
# On the EC2 instance, after SSH-ing in:
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
# log out and back in for group membership to take effect

git clone <repo-url>
cd <repo-directory>

pip install dvc[s3]
dvc pull

docker build -t enterprise-mlops-yolo .
docker run -d -p 8000:8000 --restart unless-stopped --name yolo-service enterprise-mlops-yolo
```

`--restart unless-stopped` ensures the service automatically recovers from a container crash or instance reboot without manual intervention.

### Verification

```bash
curl http://<ec2-public-ip>:8000/health
curl -X POST http://<ec2-public-ip>:8000/predict -F "file=@data/test_img.jpg"
```

**Networking note:** the API port must be explicitly opened in the instance's security group (inbound rule, custom TCP, matching port, appropriately scoped source) — the container can be fully healthy and still be unreachable externally if this rule is missing.

### Cost Management

Deployed within AWS free-tier limits (`t2.micro`/`t3.micro`, 750 instance-hours/month for the first 12 months). A billing budget alert was configured immediately after account creation.

### Stack
`AWS EC2` · `SSH`

---

## Experiment Tracking (MLflow)

Each optimization configuration (PyTorch baseline, ONNX Runtime, OpenVINO FP16) is logged as a tracked MLflow run — capturing parameters, metrics, and the associated model artifact together, so the optimization decision is reproducible and comparable rather than asserted after the fact.

```bash
pip install mlflow
python track_experiments.py
mlflow ui --host 0.0.0.0 --port 5000
```

Each run logs: **parameters** (backend, precision), **metrics** (p50/p99 latency, throughput), **artifacts** (the model file(s) for that run). Runs locally against a SQLite-backed store — sufficient for a single project; a shared/remote tracking server would be the next step for multi-contributor use.

### Stack
`MLflow`

---

## Dataset & Model Versioning (DVC + S3)

Large binary artifacts — the dataset and exported model files — are deliberately kept out of git and versioned instead through DVC, backed by an S3 remote. Git tracks only small `.dvc` pointer files; the actual binary content lives in S3 and is fetched on demand.

```bash
dvc add data/
dvc add yolov8n_openvino_model/
dvc remote add -d myremote s3://<bucket-name>/dvc-store
dvc push
```

To retrieve current versioned artifacts on any machine (local dev, EC2, or a CI runner):

```bash
dvc pull
```

This is the same mechanism that supplies the model artifact to the CI/CD pipeline — CI runners start with a clean checkout and no local files, so `dvc pull` is what makes the Docker build reproducible on a machine that has never seen this project before.

### Stack
`DVC` · `AWS S3`

---

## Pipeline Orchestration (Apache Airflow)

A DAG models the full retraining lifecycle as an explicit, orchestrated sequence: **data ingestion → preprocessing → retrain trigger → evaluation gate → deploy**. Runs locally via Docker Compose — no managed cloud Airflow (e.g. AWS MWAA) needed for this scope, and no ongoing cost.

```bash
cd airflow
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```

DAG file: `airflow/dags/yolo_retrain_pipeline.py`

| Task | Purpose |
|---|---|
| `data_ingestion` | Pulls the current versioned dataset (DVC/S3) |
| `preprocessing` | Resize, augment, train/val split |
| `retrain_trigger` | Kicks off a retraining job |
| `eval` | Gates deployment on model quality vs. the current baseline |
| `deploy` | Only reached if the eval gate passes |

**Scope note:** `retrain_trigger` and the training loop are stubbed — a real retraining run requires GPU infrastructure and hours of compute outside this project's scope. What's demonstrated is the orchestration structure and lifecycle gating, which is the actual MLOps competency being shown.

### Stack
`Apache Airflow` · `Docker Compose`

---

## CI/CD Pipeline (GitHub Actions)

On every push to `main`: tests run, a Docker image is built, pushed to GitHub Container Registry (GHCR), and deployed to the live EC2 instance — closing the loop from code change to running production service.

**Workflow file:** `.github/workflows/ci-cd.yml`

| Job | What it does |
|---|---|
| `test` | Installs dependencies, runs the test suite |
| `build-and-push` | Pulls versioned model artifacts via DVC, builds the Docker image, pushes to GHCR |
| `deploy` | SSHes into the EC2 instance, pulls the new image, restarts the service |

### Required repository secrets

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Lets the CI runner authenticate to S3 for `dvc pull` |
| `EC2_HOST` | Public IP/hostname of the deployment target |
| `EC2_USER` | SSH user on the EC2 instance (typically `ubuntu`) |
| `EC2_SSH_KEY` | Private key content for SSH access |

(`GITHUB_TOKEN` is provided automatically — no manual setup needed.)

CI runners start from a clean checkout with no local files; since model artifacts are intentionally excluded from git, `dvc pull` is what makes the build reproducible on a machine that has never touched this project before.

### Stack
`GitHub Actions` · `GHCR`

---

## Observability

### Metrics & Dashboards (Prometheus + Grafana)

The service is instrumented with `prometheus_client`, exposing:

- **Request volume** — `predict_requests_total` (counter, labeled by `status`)
- **Latency** — `predict_latency_seconds` (histogram, enabling p50/p99 queries)
- **Drift signal** — `input_brightness_drift_zscore` (gauge, see below)

Prometheus and Grafana run locally via Docker Compose, scraping the deployed service's `/metrics` endpoint over the network — deliberately decoupled from the EC2 instance itself so the small free-tier host only runs the model service, not the monitoring stack.

```bash
cd monitoring
docker compose up -d
```

Example Grafana panel queries:
- Request volume: `rate(predict_requests_total[1m])`
- p50 latency: `histogram_quantile(0.50, rate(predict_latency_seconds_bucket[5m]))`
- p99 latency: `histogram_quantile(0.99, rate(predict_latency_seconds_bucket[5m]))`
- Error rate: `rate(predict_requests_total{status="error"}[1m])`

### Load Testing (Locust)

A sustained load test validated the deployment's behavior under concurrent traffic:

```bash
locust -f locustfile.py --host http://<deployment-host>:8000 \
  --users 20 --spawn-rate 5 --run-time 3m --headless
```

**Results — 20 concurrent users, 3-minute sustained run, `POST /predict`:**

| Metric | Value |
|---|---|
| Total requests | 1,335 |
| Failures | 0 (0.00%) |
| Sustained throughput | 7.51 req/s |
| p50 latency | 570 ms |
| p95 latency | 990 ms |
| p99 latency | 1,400 ms |

**Zero failures across the full run** — the service held up cleanly at this concurrency level. Latency under load (p50 570ms) is substantially higher than the single-request benchmark (p50 29ms) — an honest and expected finding, not a contradiction: the free-tier instance has limited, burstable vCPU capacity, so concurrent requests queue and contend for CPU time. This is a genuine capacity-ceiling data point, reported as such rather than hidden behind the best-case single-request number.

**Estimated cost per inference** (using sustained throughput, assuming `t3.small` on-demand pricing of ~$0.0208/hr):

```
($0.0208 / 3600) / 7.51 req/s ≈ $0.00000077 per inference
≈ $0.77 per 1 million inferences
```

### Drift Detection

A lightweight, interpretable proxy for input distribution shift: each incoming image's mean pixel brightness is compared against a baseline computed from the reference dataset, producing a z-score.

```python
z_score = (current_image_mean_brightness - BASELINE_MEAN) / BASELINE_STD
```

The baseline (`BASELINE_MEAN_BRIGHTNESS`, `BASELINE_STD_BRIGHTNESS`) is computed once from the reference dataset via `compute_baseline_stats.py`, not hardcoded.

**Validation — before/after:**

| Input | Drift z-score | Interpretation |
|---|---|---|
| Reference-distribution image | 0.0008 | Matches training distribution — no drift |
| Out-of-distribution image | 3.86 | ~4 standard deviations out — strong, clearly-flaggable shift |

**Scope note, stated plainly:** this is a single-dimension proxy signal (brightness), not a rigorous multivariate drift detector — a production system would extend this with something like population stability index or an embedding-based OOD detector. It's also worth noting explicitly that drift and correctness are different things: a drift-flagged input can still be predicted correctly (and vice versa) — the z-score signals *distributional* difference from the reference set, not model failure.

### Stack
`Prometheus` · `Grafana` · `Locust` · `prometheus_client`

---

## Resume Bullets

- Optimized a YOLOv8n object detection model for CPU-based edge inference using OpenVINO, achieving a **65% reduction in p50 latency** and a **2.6x throughput improvement** over PyTorch baseline
- Deployed a containerized ML serving pipeline to AWS EC2 with automated CI/CD (GitHub Actions → GHCR → SSH deploy), reducing manual deployment steps and enabling crash/reboot-safe recovery via container restart policies
- Built a versioned, orchestrated ML lifecycle — DVC/S3 for dataset and model artifact versioning, Apache Airflow for pipeline orchestration (ingest → preprocess → retrain → eval-gate → deploy), and MLflow for experiment tracking
- Instrumented the service with Prometheus/Grafana observability and validated production readiness via sustained Locust load testing (**20 concurrent users, 0% failure rate, 7.51 req/s sustained**) and a validated lightweight drift-detection signal (z-score 0.0008 in-distribution vs. 3.86 out-of-distribution)
