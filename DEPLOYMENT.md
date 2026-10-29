# Deployment & Operations Guide

## Overview

This repository provides an end-to-end medical deep learning web application for 3D multi-modal Brain MRI classification and volumetric segmentation.

- **Frontend:** Single-page application (`frontend/index.html`) using vanilla JavaScript, modern CSS, and an embedded offline NiiVue WebGL medical viewer (`frontend/niivue.umd.js`).
- **Backend:** Flask REST API (`api/app.py` & `wsgi.py`) orchestrating hierarchical Stage 1 / Stage 2 classifiers and PyTorch/MONAI 3D Residual U-Net segmentation models.
- **Reporting & Exports:** Vector PDF medical reports (`api/pdf_report.py` via ReportLab), 2D annotated axial cross-section PNGs, and tidy volumetric CSVs (`api/export_utils.py`).

---

## 1. System Requirements & Hardware Considerations

### CPU vs. GPU Inference Limitations
- **GPU (NVIDIA CUDA):**
  - Recommended for interactive clinical research responsiveness.
  - Full-volume sliding-window inference with Gaussian blending (ROI: $64 \times 64 \times 64$, overlap: 0.50) executes in approximately **5–12 seconds** on an NVIDIA RTX 3060 / T4 or higher.
  - VRAM Requirement: $\ge 6\text{ GB}$.
- **CPU (Fallback):**
  - Fully supported via PyTorch native CPU tensors.
  - Full-volume sliding-window inference takes approximately **60–120 seconds** per 4-channel volume ($240 \times 240 \times 155$ voxels).
  - System RAM Requirement: **Minimum 4 GB, recommended 8 GB+** to hold resampled multi-modal arrays and intermediate probability volumes without triggering Out-Of-Memory (OOM) fatal kills.
- **Serverless Warning (Vercel / AWS Lambda):**
  - Standard serverless runtimes have hard execution timeouts (10–60 seconds) and package size limits (~250 MB). Full-volume 3D PyTorch segmentation will trigger execution timeouts or memory crashes on generic serverless endpoints unless delegated to an asynchronous GPU task queue (e.g. Celery + Redis + RunPod / Modal).

---

## 2. Model Checkpoint & MRI Data Storage Strategy

> **Crucial Rule:** Model weights (`.pth` / `.pt`) and raw patient MRI volumes (`.nii` / `.nii.gz`) must **never** be committed to ordinary Git history due to repository bloat, privacy considerations, and Git transfer quotas.

### Recommended Storage Strategies:
1. **Cloud Object Storage (GCS / AWS S3 / Cloudflare R2):**
   - Store versioned weights in a private object storage bucket:
     - `gs://your-bucket/weights/glioma_best_medium_experiment.pth`
     - `gs://your-bucket/weights/meningioma_best.pth`
     - `gs://your-bucket/weights/classifier_stage1.pth`
     - `gs://your-bucket/weights/classifier_stage2.pth`
   - Add a pre-run hydration script (e.g., `python scripts/download_weights.py`) in your CI/CD or Docker container startup sequence to download missing weights if not cached locally.
2. **Hugging Face Model Hub / Git LFS:**
   - Host checkpoint binaries on Hugging Face Hub under a restricted model repository and pull via `huggingface_hub.hf_hub_download`.
3. **Persistent Volume Mounts (Docker / Kubernetes):**
   - For containerized deployments, mount a persistent volume containing the model weights directly to the container's `/app/outputs/` directory.

---

## 3. Local Production Execution

For Windows and cross-platform production serving without dev-server warnings:

### Using Waitress (Windows / Linux / macOS)
```powershell
# Activate your virtual environment
.\.venv\Scripts\activate

# Install production runner
pip install waitress

# Launch production server on port 5000
waitress-serve --listen=0.0.0.0:5000 wsgi:application
```

### Using Gunicorn (Linux / macOS)
```bash
source .venv/bin/activate
pip install gunicorn

# Run with 1 worker and 4 threads (to prevent PyTorch memory multiplication)
gunicorn wsgi:application \
  --bind 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 180 \
  --access-logfile -
```

---

## 4. Cloud Deployment (Render Blueprint)

A declarative `render.yaml` specification is included in the project root:

1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Blueprint** service selecting `render.yaml`.
3. Choose at least a **Standard (2GB RAM)** instance type to accommodate multi-channel 3D volume loading.
4. Ensure environment variables (`TORCH_DEVICE=cpu`, `FLASK_ENV=production`) are populated.
5. In Render, request timeouts should be set to $\ge 180$ seconds to allow CPU sliding-window inference to complete cleanly.

---

## 5. Docker Deployment

### Sample `Dockerfile`
```dockerfile
FROM python:3.12-slim

# Install system dependencies for SimpleITK and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application source code
COPY api/ api/
COPY frontend/ frontend/
COPY integration/ integration/
COPY models/ models/
COPY preprocessing/ preprocessing/
COPY segmentation/ segmentation/
COPY wsgi.py .
COPY render.yaml .

# Create output run directory
RUN mkdir -p outputs/api_runs

EXPOSE 5000

ENV FLASK_ENV=production
ENV TORCH_DEVICE=cpu

CMD ["gunicorn", "wsgi:application", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180"]
```

---

## 6. Pre-flight Smoke Test & Health Check

After deployment, verify the service by requesting the health check endpoint:

```bash
curl -X GET http://localhost:5000/api/health
```

Expected JSON Response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "device_name": "Intel/AMD CPU",
  "glioma_model_loaded": true,
  "meningioma_model_loaded": true,
  "automatic_classification_available": true
}
```
