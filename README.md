# Face Clustering API

> **Take-home assignment for Hung Hing Printing - AI Science Team**
>
> Deploy a face clustering algorithm as a containerized API service.

---

## Overview

This project wraps a face clustering algorithm into a production-ready REST API using **FastAPI** and **Docker**. Given a set of event photos, the API detects all faces, extracts 128-dimensional embeddings using `dlib`, clusters them with **DBSCAN**, and returns a cluster assignment for each face.

### Key Features

- **Multi-image upload**: Process multiple event photos in a single request
- **Configurable clustering**: Adjust DBSCAN parameters (`eps`, `min_samples`) and detection model (`hog`/`cnn`)
- **Structured JSON output**: Each face gets a unique ID, cluster assignment, bounding box, and source image
- **Health monitoring**: Built-in health check endpoint for container orchestration
- **Interactive API docs**: Auto-generated Swagger UI at `/docs`
- **Security**: Non-root user, input validation, SQL injection prevention

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| Face Detection | `face_recognition` (dlib 128-d embeddings) |
| Clustering | DBSCAN (scikit-learn) |
| Image Processing | OpenCV |
| Container | Docker (multi-stage build) |
| Python | 3.9 |

---

## Architecture

```
Client Request (multipart/form-data with images)
        │
        ▼
┌─────────────────────────────────────┐
│  POST /cluster                      │
│  ├─ Input validation                │
│  ├─ Image decoding (bytes → RGB)    │
│  └─ Parameter parsing (eps, etc.)   │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │  Face Detection  │  face_recognition.face_locations()
        │  (HOG or CNN)    │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Embedding    │  face_recognition.face_encodings()
        │  (128-d vec)  │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  DBSCAN       │  sklearn.cluster.DBSCAN
        │  Clustering   │  eps, min_samples configurable
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  JSON Response│  faces[] + clusters[] + metadata
        └─────────────┘
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Git](https://git-scm.com/downloads)
- (Optional) Python 3.9+ for local client testing

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-clustering-api.git
cd face-clustering-api
```

### 2. Build Docker Image

```bash
docker build -t face-clustering-api .
```

> **Note**: The first build takes ~10-15 minutes because `dlib` must be compiled from source. Subsequent builds are faster due to Docker layer caching.

### 3. Run Container

```bash
docker run -d -p 8000:8000 --name face-clustering face-clustering-api
```

### 4. Verify Service

```bash
# Health check
curl http://localhost:8000/health

# Service info
curl http://localhost:8000/info
```

Expected response:
```json
{
  "status": "healthy",
  "service": "face-clustering-api",
  "version": "1.0.0"
}
```

### 5. Test Clustering API

Upload images and get cluster assignments:

```bash
# Single image
curl -X POST "http://localhost:8000/cluster" \
  -F "images=@/path/to/photo1.jpg" \
  -F "images=@/path/to/photo2.jpg" \
  -F "eps=0.4" \
  -F "min_samples=2"
```

Or use the provided Python client:

```bash
pip install requests
python client_example/test_client.py --images /path/to/event_photos/
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/info` | Service information |
| `POST` | `/cluster` | Upload images and cluster faces |
| `GET` | `/docs` | Swagger UI (interactive docs) |
| `GET` | `/redoc` | ReDoc (alternative docs) |

### POST /cluster

Upload images for face clustering.

**Parameters (form-data):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `images` | File | Yes | - | One or more image files (jpg, png, bmp, gif, webp) |
| `eps` | float | No | 0.4 | DBSCAN epsilon (0.1-1.0). Lower = stricter matching |
| `min_samples` | int | No | 2 | Min faces to form a cluster (1-10) |
| `detection_model` | string | No | hog | `hog` (CPU, fast) or `cnn` (GPU, accurate) |

**Response (JSON):**

```json
{
  "status": "success",
  "message": "Clustering complete. Found 3 unique people from 12 faces.",
  "total_images": 5,
  "total_faces": 12,
  "num_clusters": 3,
  "num_noise": 1,
  "processing_time_ms": 2345.67,
  "parameters": {
    "eps": 0.4,
    "min_samples": 2,
    "detection_model": "hog"
  },
  "faces": [
    {
      "face_id": 0,
      "cluster_id": 0,
      "image_filename": "event_001.jpg",
      "bounding_box": {
        "top": 100,
        "right": 200,
        "bottom": 300,
        "left": 50
      }
    }
  ],
  "clusters": [
    {
      "cluster_id": 0,
      "face_count": 4,
      "face_ids": [0, 1, 2, 3]
    },
    {
      "cluster_id": -1,
      "face_count": 1,
      "face_ids": [11]
    }
  ]
}
```

**Cluster ID meanings:**

| Cluster ID | Meaning |
|-----------|---------|
| `0, 1, 2, ...` | Unique person clusters |
| `-1` | Noise / unclustered (single-occurrence or poor quality faces) |

---

## Docker Commands

```bash
# Build image
docker build -t face-clustering-api .

# Run container (foreground)
docker run -p 8000:8000 face-clustering-api

# Run container (detached)
docker run -d -p 8000:8000 --name face-clustering face-clustering-api

# View logs
docker logs -f face-clustering

# Stop container
docker stop face-clustering

# Remove container
docker rm face-clustering

# Save image to tar (for submission)
docker save face-clustering-api > face-clustering-api.tar

# Load image from tar
docker load < face-clustering-api.tar
```

### Using Docker Compose

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

---

## Project Structure

```
face-clustering-api/
├── app/
│   ├── __init__.py              # Package init
│   ├── main.py                  # FastAPI application & endpoints
│   ├── face_clustering.py       # Core clustering engine
│   ├── models.py                # Pydantic request/response models
│   └── config.py                # Configuration settings
├── tests/
│   ├── __init__.py
│   └── test_api.py              # Automated API tests
├── client_example/
│   ├── test_client.py           # Python client for calling API
│   └── curl_examples.sh         # cURL command examples
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── .dockerignore                # Docker build exclusions
└── README.md                    # This file
```

---

## Algorithm Details

### Face Detection
- **Library**: `face_recognition` (wraps dlib's CNN/HOG detectors)
- **Output**: Bounding box `(top, right, bottom, left)` per face
- **Models**: `hog` (CPU-optimized) or `cnn` (GPU-accelerated, more accurate)

### Face Embedding
- **Method**: 128-dimensional embedding via dlib's pre-trained ResNet model
- **Property**: Faces of the same person are close in Euclidean space
- **Normalization**: L2-normalized vectors (standard for face recognition)

### Clustering
- **Algorithm**: DBSCAN (Density-Based Spatial Clustering)
- **Advantage**: No need to pre-specify number of people; discovers clusters automatically
- **Distance**: Euclidean distance on 128-d embeddings
- **Noise handling**: Outliers marked as `-1` (unclustered)

### Parameter Tuning

| Parameter | Default | Lower | Higher | Effect |
|-----------|---------|-------|--------|--------|
| `eps` | 0.4 | 0.35 | 0.5 | Stricter vs looser face matching |
| `min_samples` | 2 | 1 | 5+ | Min photos per person to form cluster |
| `detection_model` | hog | - | cnn | Speed vs accuracy tradeoff |

---

## Testing

### Run Automated Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py -v --cov=app
```

### Manual Testing with cURL

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Upload single image
curl -X POST "http://localhost:8000/cluster" \
  -F "images=@sample.jpg"

# 3. Upload multiple images with custom parameters
curl -X POST "http://localhost:8000/cluster" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg" \
  -F "eps=0.35" \
  -F "min_samples=2" \
  -F "detection_model=hog"
```

### Manual Testing with Python Client

```bash
# Check API health
python client_example/test_client.py --health

# Cluster images in directory
python client_example/test_client.py --images ./event_photos/

# Cluster with custom parameters
python client_example/test_client.py --images ./photos/ --eps 0.35 --min-samples 3

# Save results to file
python client_example/test_client.py --images ./photos/ --save results.json
```

---

## Test Results

### API Endpoint Tests

| Test Case | Method | Expected | Result |
|-----------|--------|----------|--------|
| Health check | GET /health | 200 OK, status=healthy | ✅ PASS |
| Service info | GET /info | 200 OK, endpoints listed | ✅ PASS |
| Empty request | POST /cluster | 400 Bad Request | ✅ PASS |
| Single image upload | POST /cluster | 200 OK, faces detected | ✅ PASS |
| Multiple image upload | POST /cluster | 200 OK, all processed | ✅ PASS |
| Custom parameters | POST /cluster (eps=0.35) | 200 OK, params reflected | ✅ PASS |
| Invalid eps (2.0) | POST /cluster | 422 Validation Error | ✅ PASS |
| Invalid min_samples (0) | POST /cluster | 422 Validation Error | ✅ PASS |
| Non-image file | POST /cluster | 200 OK, 0 faces (graceful) | ✅ PASS |
| Swagger UI | GET /docs | 200 OK, interactive docs | ✅ PASS |

### Docker Tests

| Test Case | Command | Result |
|-----------|---------|--------|
| Image build | `docker build` | ✅ SUCCESS (multi-stage) |
| Container start | `docker run` | ✅ HEALTHY |
| Port mapping | `curl localhost:8000/health` | ✅ REACHABLE |
| Health endpoint | `docker exec` healthcheck | ✅ PASS |
| Graceful shutdown | `docker stop` | ✅ CLEAN EXIT |

---

## Improvements & Optimizations

### Implemented
- [x] Multi-stage Docker build for smaller image size
- [x] Non-root user for container security
- [x] Input validation with Pydantic models
- [x] Comprehensive error handling
- [x] Health check endpoint for orchestration
- [x] Interactive API documentation (Swagger UI)
- [x] Configurable clustering parameters
- [x] Structured JSON response with cluster summaries

### Suggested Future Improvements

| Priority | Improvement | Impact |
|----------|-------------|--------|
| **High** | **GPU support** (NVIDIA runtime) | 10x speedup for CNN model |
| **High** | **Async image processing** (background tasks) | Handle larger batches without timeout |
| **Medium** | **Caching embeddings** (Redis/SQLite) | Skip re-processing same images |
| **Medium** | **Batch result persistence** (DB/S3) | Store results for later retrieval |
| **Medium** | **Face quality scoring** | Filter blurry/low-quality detections |
| **Low** | **Hierarchical clustering** (Agglomerative) | Better for large events (100+ people) |
| **Low** | **WebSocket streaming** | Real-time progress for large batches |
| **Low** | **Metrics endpoint** (Prometheus) | Monitor clustering performance |

### Performance Notes

- **HOG model**: ~1-2 seconds per image on CPU (sufficient for batch processing)
- **CNN model**: ~0.5 seconds per image on GPU (requires NVIDIA runtime)
- **Recommended max batch size**: 50 images per request
- **Memory usage**: ~500MB base + ~50MB per 10 concurrent images

---

## Submission Checklist

- [x] **Docker image** built and tested
- [x] **Source code** on GitHub with version control
- [x] **API client code** for local testing (`client_example/`)
- [x] **Documentation** (this README with API reference)
- [x] **Test results** documented (automated + manual)
- [x] **Error handling** validated (invalid input, edge cases)
- [x] **Improvement suggestions** provided

---

## License

This project is created as a technical assignment for Hung Hing Printing.
