"""
Face Clustering API - FastAPI Application

Provides REST endpoints for clustering faces in uploaded images.
"""
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn

from app.face_clustering import process_images
from app.models import (
    ClusteringResponse,
    HealthResponse,
    InfoResponse,
    DetectionModel,
    ClusterConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "Face Clustering API"
APP_DESCRIPTION = """
API service for clustering faces in event photos.

Upload multiple images and the service will:
1. Detect all faces in each image
2. Extract 128-dimensional face embeddings
3. Cluster faces using DBSCAN algorithm
4. Return cluster assignments for each face

**Usage:**
- Send a POST request to `/cluster` with image files
- Receive JSON response with face IDs and their cluster assignments
- Cluster ID -1 indicates unclustered/noise faces
"""
APP_VERSION = "1.0.0"

# Create FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status to verify the API is running.
    """
    return HealthResponse()


@app.get("/info", response_model=InfoResponse, tags=["System"])
async def service_info():
    """
    Service information endpoint.
    
    Returns API metadata, available endpoints, and configuration details.
    """
    return InfoResponse()


@app.post(
    "/cluster",
    response_model=ClusteringResponse,
    tags=["Clustering"],
    responses={
        400: {"description": "Bad request - invalid input"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)
async def cluster_faces(
    images: List[UploadFile] = File(
        ...,
        description="One or more image files to process (jpg, png, bmp, gif, webp)"
    ),
    eps: float = Form(
        default=0.4,
        ge=0.1,
        le=1.0,
        description="DBSCAN epsilon: max distance for neighborhood (lower = stricter matching)"
    ),
    min_samples: int = Form(
        default=2,
        ge=1,
        le=10,
        description="Minimum faces to form a cluster"
    ),
    detection_model: DetectionModel = Form(
        default=DetectionModel.HOG,
        description="Face detection model: 'hog' (faster, CPU) or 'cnn' (accurate, needs GPU)"
    )
):
    """
    Face clustering endpoint.
    
    Upload multiple images and get cluster assignments for each detected face.
    
    **Parameters:**
    - **images**: Image files to process (required, 1-50 files recommended)
    - **eps**: Clustering strictness (0.35=strict, 0.4=default, 0.5=lenient)
    - **min_samples**: Min faces per cluster (default: 2, set to 1 for single-photo grouping)
    - **detection_model**: 'hog' for CPU, 'cnn' for GPU (if available)
    
    **Returns:**
    - Cluster assignments for each face
    - Bounding box coordinates
    - Cluster statistics
    - Processing metadata
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "message": "Clustering complete. Found 3 unique people from 12 faces.",
        "total_images": 5,
        "total_faces": 12,
        "num_clusters": 3,
        "num_noise": 1,
        "processing_time_ms": 2345.67,
        "parameters": {"eps": 0.4, "min_samples": 2, "detection_model": "hog"},
        "faces": [...],
        "clusters": [...]
    }
    ```
    """
    # Validate input
    if not images:
        raise HTTPException(status_code=400, detail="No images provided. Upload at least one image.")
    
    if len(images) > 100:
        raise HTTPException(status_code=400, detail="Too many images. Maximum 100 files per request.")
    
    logger.info(f"Processing {len(images)} images with eps={eps}, min_samples={min_samples}, model={detection_model}")
    
    # Read uploaded files into memory
    uploaded_files = []
    for upload_file in images:
        try:
            content = await upload_file.read()
            if len(content) == 0:
                logger.warning(f"Empty file: {upload_file.filename}")
                continue
            uploaded_files.append((upload_file.filename, content))
        except Exception as e:
            logger.error(f"Error reading file {upload_file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading file {upload_file.filename}: {str(e)}")
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid image files found.")
    
    try:
        # Process images through clustering pipeline
        result = process_images(
            uploaded_files=uploaded_files,
            eps=eps,
            min_samples=min_samples,
            detection_model=detection_model.value
        )
        
        logger.info(
            f"Clustering complete: {result['num_clusters']} clusters, "
            f"{result['total_faces']} faces, {result['processing_time_ms']}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clustering processing failed: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions gracefully"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Internal server error: {str(exc)}"
        }
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
