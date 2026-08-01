# Production ML Serving Platform — Edge Vision Model

A production-oriented serving pipeline for a YOLOv8n object detection model, demonstrating end-to-end ML lifecycle ownership: optimization → packaging → containerized deployment → cloud deployment → automated pipeline & CI/CD → (upcoming) observability.

## Business Problem

Most ML models never leave the notebook they were trained in — or if they do, they run too slowly, too expensively, or with no visibility into whether they're still trustworthy once deployed. This project addresses the core production-readiness gap for CPU-bound edge deployment: taking a trained CV model and making it fast enough to serve in real time, portable enough to run anywhere, versioned and orchestrated like a real system, and instrumented enough to trust once it's live.

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
curl -X POST http://localhost:8000/predict -F "file=@test_img.jpg"
```

### Run the orchestration pipeline locally

```bash
cd airflow
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```

Open `http://localhost:8080` (default login `airflow` / `airflow`), and the `yolo_retrain_pipeline` DAG will be listed and runnable from the UI.

### Deploy your own copy

See [Cloud Deployment](#cloud-deployment-aws-ec2) and [CI/CD Pipeline](#cicd-pipeline-github-actions) below for full setup, including the repo secrets required for automated deployment.

---

## Model Optimization & Containerized Serving

### Objective
Convert a baseline PyTorch model into an optimized, containerized inference service suitable for CPU-based edge deployment, with benchmarked evidence of the performance gain.

### Approach

1. **Baseline benchmark** — measured end-to-end inference latency (including preprocessing/postprocessing) for the unoptimized PyTorch model.
2. **Export & optimize** — converted the model to ONNX and to OpenVINO IR format (FP16), Intel's inference runtime purpose-built for CPU deployment.
3. **Apples-to-apples comparison** — benchmarked all three backends (PyTorch, ONNX Runtime, OpenVINO) through an identical harness: same image, same machine, same warmup protocol, 100 timed iterations each.
4. **API wrapper** — built a FastAPI service exposing `/predict` and `/health` endpoints, with the OpenVINO model as the serving backend.
5. **Containerization** — packaged the service in Docker for portable, reproducible deployment.

### Results

Benchmarked on an Intel 13th-gen i7 (CPU-only, no GPU), 100 iterations per backend:

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

Current image size: 1.28GB. This includes `ultralytics`/`torch`, which were only required for the export step, not for serving — a multi-stage build separating export-time and serving-time dependencies is a planned optimization to reduce this footprint.

### Stack
`Python` · `PyTorch` · `ONNX` · `OpenVINO` · `FastAPI` · `OpenCV` · `Docker`

---

## Cloud Deployment (AWS EC2)

### Objective
Move the containerized service from local-only execution to a publicly reachable cloud deployment.

### Infrastructure
- Ubuntu Server (22.04/24.04 LTS), EC2 free-tier-eligible instance type
- Docker installed directly on the instance
- Security group configured to allow inbound traffic on the API port from the required source

### Setup steps

```bash
# On the EC2 instance, after SSH-ing in:
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
# log out and back in for group membership to take effect

git clone <repo-url>
cd <repo-directory>

# pull versioned model artifacts from the S3-backed DVC remote
pip install dvc[s3]
dvc pull

docker build -t enterprise-mlops-yolo .
docker run -d -p 8000:8000 --restart unless-stopped --name yolo-service enterprise-mlops-yolo
```

The `--restart unless-stopped` policy ensures the service automatically recovers from a container crash or an instance reboot without manual intervention — a baseline reliability guarantee for anything claiming to be production-facing.

### Verification

From an external machine (confirming the deployment is actually publicly reachable, not just alive on localhost):

```bash
curl http://<ec2-public-ip>:8000/health
curl -X POST http://<ec2-public-ip>:8000/predict -F "file=@test_img.jpg"
```

**Networking note:** the API port must be explicitly opened in the instance's security group (inbound rule, custom TCP, matching port, appropriately scoped source) — this is a common first-deployment stumbling block, since the container can be fully healthy and still be unreachable externally if this rule is missing or misconfigured.

### Cost Management

Deployed entirely within AWS free-tier limits (`t2.micro`/`t3.micro`, 750 instance-hours/month for the first 12 months). A billing budget alert was configured immediately after account creation as a standard safeguard against unexpected charges.

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

Each run logs:
- **Parameters:** backend, precision
- **Metrics:** p50 latency, p99 latency, throughput
- **Artifacts:** the corresponding model file(s) for that run

This runs locally against a SQLite-backed MLflow store — sufficient for tracking a single project's experiments; a shared/remote tracking server would be the next step if multiple contributors or environments needed to log to the same registry.

### Stack
`MLflow`

---

## Dataset & Model Versioning (DVC + S3)

Large binary artifacts — the training dataset and the exported model files — are deliberately kept out of git and versioned instead through DVC, backed by an S3 remote. Git only ever tracks small `.dvc` pointer files (hashes/metadata); the actual binary content lives in S3 and is fetched on demand.

```bash
dvc add data/
dvc add yolov8n_openvino_model/
dvc remote add -d myremote s3://<bucket-name>/dvc-store
dvc push
```

To retrieve the current versioned artifacts on any machine (local dev, EC2, or a CI runner):

```bash
dvc pull
```

This is the same mechanism that supplies the model artifact to the CI/CD pipeline below — CI runners start with a clean checkout and no local files, so `dvc pull` is what makes the Docker build reproducible on a machine that's never seen this project before.

### Stack
`DVC` · `AWS S3`

---

## Pipeline Orchestration (Apache Airflow)

A DAG models the full retraining lifecycle as an explicit, orchestrated sequence rather than an ad hoc script: **data ingestion → preprocessing → retrain trigger → evaluation gate → deploy**.

Runs locally via Docker Compose — no managed cloud Airflow (e.g. AWS MWAA) needed for this scope, and no ongoing cost.

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

**Scope note:** `retrain_trigger` and the underlying training loop are stubbed rather than fully implemented — a real retraining run requires GPU infrastructure and hours of compute outside this project's scope. What's being demonstrated here is the orchestration structure and lifecycle gating (ingest → preprocess → retrain → evaluate → conditionally deploy), which is the actual MLOps competency, independent of how long the stubbed training step would take in a fuller implementation.

### Stack
`Apache Airflow` · `Docker Compose`

---

## CI/CD Pipeline (GitHub Actions)

On every push to `main`, the pipeline runs tests, builds a Docker image, pushes it to GitHub Container Registry (GHCR), and deploys it to the live EC2 instance — closing the loop from code change to running production service.

**Workflow file:** `.github/workflows/ci-cd.yml`

| Job | What it does |
|---|---|
| `test` | Installs dependencies, runs the test suite |
| `build-and-push` | Pulls versioned model artifacts via DVC, builds the Docker image, pushes to GHCR |
| `deploy` | SSHes into the EC2 instance, pulls the new image, and restarts the service |

### Required repository secrets

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Lets the CI runner authenticate to S3 for `dvc pull` |
| `EC2_HOST` | Public IP/hostname of the deployment target |
| `EC2_USER` | SSH user on the EC2 instance (typically `ubuntu`) |
| `EC2_SSH_KEY` | Private key content for SSH access |

(`GITHUB_TOKEN` is provided automatically by GitHub Actions — no manual setup needed.)

### Why the CI runner needs `dvc pull`

CI runners start from a clean checkout with no local files. Since model artifacts are intentionally excluded from git, the build would fail without first retrieving them from the S3-backed DVC remote — the same versioning system used for local development and EC2 deployment, now also serving as the mechanism that makes automated builds reproducible on a machine that's never touched this project before.

### Stack
`GitHub Actions` · `GHCR`

