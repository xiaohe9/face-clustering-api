# Face Clustering API - Technical Documentation

> This document explains the architecture, data flow, and deployment process in detail.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Flow](#data-flow)
- [File Structure](#file-structure)
- [Core Algorithm](#core-algorithm)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Nginx Reverse Proxy](#nginx-reverse-proxy)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
Client (Browser/cURL/Python)
    |
    | HTTP Request
    v
+-----------------------------+
|   Docker Container          |
|   +---------------------+   |
|   |  Uvicorn (port 8000)|   |
|   |  +---------------+  |   |
|   |  | FastAPI App   |  |   |
|   |  | - /health     |  |   |
|   |  | - /info       |  |   |
|   |  | - /cluster    |  |   |
|   |  +-------+-------+  |   |
|   +--------|------------+   |
+------------|----------------+
             |
             v
+-----------------------------+
|  Face Clustering Engine     |
|  (face_clustering.py)       |
|                             |
|  1. Face Detection          |
|     face_recognition        |
|     .face_locations()       |
|              |              |
|              v              |
|  2. Feature Extraction      |
|     128-d embedding         |
|     .face_encodings()       |
|              |              |
|              v              |
|  3. DBSCAN Clustering       |
|     eps=0.4, min_samples=2  |
|              |              |
|              v              |
|  4. Result Formatting       |
|     JSON response           |
+-----------------------------+
```

---

## Data Flow

### Step-by-Step: What Happens When You Upload Images

```
[Client]                    [Server]
   |                            |
   |  POST /cluster             |
   |  Content-Type: multipart   |
   |  (image files)             |
   |--------------------------->|
   |                            |
   |                     [FastAPI]
   |                       1. Receive files
   |                       2. Validate (images only)
   |                       3. Pass to engine
   |                            |
   |                     [Engine]
   |                       For each image:
   |                         a. Decode bytes -> RGB array
   |                         b. Detect faces -> bounding boxes
   |                         c. Extract 128-d embeddings
   |                       |
   |                       For all embeddings:
   |                         d. DBSCAN clustering
   |                         e. Assign cluster IDs
   |                            |
   |                       [JSON Response]
   |                       {                              |
   |  <-----------------------  "total_faces": 12,       |
   |  HTTP 200 OK              "num_clusters": 3,       |
   |                           "faces": [                 |
   |                             {                        |
   |                               "face_id": 0,          |
   |                               "cluster_id": 0,       |
   |                               "bbox": {...}          |
   |                             }                        |
   |                           ]                          |
   |                       }                              |
```

---

## File Structure

```
face-clustering-api/
|
|-- app/                          # Application source code
|   |-- __init__.py               # Package marker
|   |-- main.py                   # FastAPI entry point
|   |   - Creates FastAPI app
|   |   - Defines 3 API endpoints
|   |   - Handles request/response
|   |
|   |-- face_clustering.py        # Core algorithm
|   |   - load_image_from_bytes() # Decode uploaded files
|   |   - detect_and_encode()     # Face detection + embeddings
|   |   - cluster_embeddings()    # DBSCAN clustering
|   |   - process_images()        # Main pipeline orchestrator
|   |
|   |-- models.py                 # Data validation schemas
|       - ClusterConfig           # Request parameters
|       - ClusteringResponse      # Response structure
|       - FaceResult              # Single face result
|
|-- client_example/               # Client code for testing
|   |-- test_client.py            # Python client (requests)
|   |-- curl_examples.sh          # Shell/cURL examples
|
|-- tests/                        # Automated tests
|   |-- test_api.py               # pytest test cases
|
|-- Dockerfile                    # Multi-stage Docker build
|-- docker-compose.yml            # One-command deployment
|-- requirements.txt              # Python dependencies
|-- .dockerignore                 # Docker build exclusions
|-- .gitignore                    # Git exclusions
|-- README.md                     # User documentation
|-- TECHNICAL.md                  # This file
```

---

## Core Algorithm

### 1. Face Detection

**Library:** `face_recognition` (wraps dlib)

**Input:** RGB image array
**Output:** List of bounding boxes `(top, right, bottom, left)`

```python
# HOG model = CPU-based, fast
# CNN model = GPU-based, more accurate
boxes = face_recognition.face_locations(image, model="hog")
```

**What it does:**
- Scans the image for face-like patterns
- Returns coordinates of each detected face
- Ignores non-face regions

### 2. Feature Extraction (Embedding)

**Input:** Image + bounding boxes
**Output:** 128-dimensional vector per face

```python
encodings = face_recognition.face_encodings(image, boxes)
# Returns: array of shape (N, 128) where N = number of faces
```

**Key property:**
- Same person -> vectors are close in Euclidean space
- Different people -> vectors are far apart
- These vectors are L2-normalized (length = 1)

### 3. DBSCAN Clustering

**Algorithm:** Density-Based Spatial Clustering
**Input:** N x 128 matrix of face embeddings
**Output:** Cluster labels (-1 = noise)

```python
clt = DBSCAN(eps=0.4, min_samples=2, metric="euclidean")
labels = clt.fit_predict(embeddings)
```

**How it works:**
1. Pick a point, find all neighbors within distance `eps`
2. If there are at least `min_samples` neighbors, form a cluster
3. Expand cluster by adding connected points
4. Points that don't belong to any cluster = noise (-1)

**Why DBSCAN for faces?**
- **No need to specify cluster count** (we don't know how many people in photos)
- **Handles noise** (blurry faces, false detections get -1)
- **Automatic** (just set eps and min_samples)

### 4. Parameter Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `eps` | 0.4 | Max distance to be "same person". Lower = stricter matching |
| `min_samples` | 2 | Min faces to form a cluster. Set to 1 to allow single-photo people |
| `detection_model` | hog | `hog` (CPU, ~1s/img) or `cnn` (GPU, ~0.5s/img, more accurate) |

---

## API Endpoints

### `GET /health`

**Purpose:** Check if service is running

**Response:**
```json
{
  "status": "healthy",
  "service": "face-clustering-api",
  "version": "1.0.0"
}
```

**Use case:** Docker healthcheck, monitoring

---

### `GET /info`

**Purpose:** Get service metadata

**Response:**
```json
{
  "service": "Face Clustering API",
  "version": "1.0.0",
  "endpoints": {
    "POST /cluster": "Upload images and get face cluster assignments"
  },
  "clustering_algorithm": "DBSCAN",
  "embedding_model": "dlib_face_recognition (128-d)"
}
```

---

### `POST /cluster`

**Purpose:** Upload images, get face clustering results

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `images` | File | Yes | - | One or more image files |
| `eps` | float | No | 0.4 | Clustering distance threshold (0.1-1.0) |
| `min_samples` | int | No | 2 | Min faces per cluster (1-10) |
| `detection_model` | string | No | hog | `hog` or `cnn` |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/cluster" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "eps=0.4"
```

**Response:**
```json
{
  "status": "success",
  "message": "Found 3 unique people from 12 faces",
  "total_images": 2,
  "total_faces": 12,
  "num_clusters": 3,
  "num_noise": 1,
  "processing_time_ms": 2345.67,
  "parameters": {"eps": 0.4, "min_samples": 2, "detection_model": "hog"},
  "faces": [
    {
      "face_id": 0,
      "cluster_id": 0,
      "image_filename": "photo1.jpg",
      "bounding_box": {"top": 100, "right": 200, "bottom": 300, "left": 50}
    }
  ],
  "clusters": [
    {"cluster_id": 0, "face_count": 4, "face_ids": [0, 1, 2, 3]},
    {"cluster_id": -1, "face_count": 1, "face_ids": [11]}
  ]
}
```

---

## Docker Deployment

### Standard Deployment

```bash
# Pull pre-built image
docker pull ghcr.io/xiaohe9/face-clustering-api:latest

# Run (port 8000)
docker run -d -p 8000:8000 --name face-clustering ghcr.io/xiaohe9/face-clustering-api:latest

# Test
curl http://localhost:8000/health
```

### Behind Nginx (Port 80)

If your server only exposes port 80 (with Nginx reverse proxy):

**Step 1:** Run Docker container on localhost only (not exposed publicly)

```bash
docker run -d -p 127.0.0.1:8000:8000 --name face-clustering ghcr.io/xiaohe9/face-clustering-api:latest
```

> Note: `127.0.0.1:8000` means only accessible from the server itself, not from the internet.

**Step 2:** Add Nginx reverse proxy config

Edit `/etc/nginx/conf.d/face-clustering.conf`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or your server IP

    location /face-clustering/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # For file uploads (increase limit if needed)
        client_max_body_size 50M;
    }
}
```

**Step 3:** Reload Nginx

```bash
nginx -t          # Test config
nginx -s reload   # Apply changes
```

**Step 4:** Access via port 80

```bash
# Health check
curl http://your-domain.com/face-clustering/health

# API docs (Swagger UI)
# Open in browser: http://your-domain.com/face-clustering/docs
```

---

## CI/CD Pipeline

**File:** `.github/workflows/docker-build.yml`

**Trigger:** Every push to `main` branch

**Flow:**
```
You push code to GitHub
        |
        v
GitHub Actions starts (Ubuntu VM)
        |
        v
Checkout code -> Login to ghcr.io
        |
        v
Docker build (multi-stage)
        |
        v
Docker push to ghcr.io
        |
        v
Image available at:
ghcr.io/xiaohe9/face-clustering-api:latest
```

**Build time:** ~5-8 minutes (dlib compilation is the bottleneck)

---

## Troubleshooting

### `ImportError: libGL.so.1`

**Cause:** OpenCV requires graphics library, slim image doesn't have it
**Fix:** Added `libgl1` to runtime stage (Dockerfile line 58)

### `failed to resolve source metadata` (Docker build)

**Cause:** Docker Hub mirror unreachable (common in China)
**Workaround:** Use GitHub Actions for remote build, or configure alternative registry mirrors

### `libgomp1` / OpenBLAS errors

**Cause:** Missing linear algebra libraries in runtime
**Fix:** Install `libopenblas0`, `liblapack3`, `libgomp1` in runtime stage

### Large Docker image size

**Cause:** dlib + face_recognition models are large
**Expected:** ~400-500MB compressed, ~1GB uncompressed
**Optimization:** Multi-stage build already minimizes this

---

## Performance Notes

| Detection Model | Hardware | Speed | Accuracy |
|-----------------|----------|-------|----------|
| HOG (default) | CPU | ~1-2s/image | Good |
| CNN | GPU (CUDA) | ~0.5s/image | Better |

**Recommended max batch size:** 50 images per request
**Memory usage:** ~500MB base + ~50MB per 10 concurrent images
