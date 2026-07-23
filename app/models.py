"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class DetectionModel(str, Enum):
    """Face detection model options"""
    HOG = "hog"      # Faster, CPU-only
    CNN = "cnn"      # More accurate, requires GPU


class BoundingBox(BaseModel):
    """Face bounding box: (top, right, bottom, left)"""
    top: int
    right: int
    bottom: int
    left: int


class ClusterConfig(BaseModel):
    """Clustering configuration parameters"""
    eps: float = Field(default=0.4, ge=0.1, le=1.0, description="DBSCAN epsilon (distance threshold)")
    min_samples: int = Field(default=2, ge=1, le=10, description="Minimum samples to form a cluster")
    detection_model: DetectionModel = Field(default=DetectionModel.HOG, description="Face detection model")


class FaceResult(BaseModel):
    """Single face clustering result"""
    face_id: int = Field(description="Unique face identifier (0-indexed)")
    cluster_id: int = Field(description="Cluster assignment (-1 = noise/unclustered)")
    image_filename: str = Field(description="Source image filename")
    bounding_box: BoundingBox = Field(description="Face bounding box coordinates")


class ClusterInfo(BaseModel):
    """Cluster summary information"""
    cluster_id: int = Field(description="Cluster identifier")
    face_count: int = Field(description="Number of faces in this cluster")
    face_ids: List[int] = Field(description="Face IDs belonging to this cluster")


class ClusteringResponse(BaseModel):
    """Main API response for face clustering"""
    status: str = Field(description="Processing status: success or error")
    message: str = Field(description="Status message")
    total_images: int = Field(description="Number of input images processed")
    total_faces: int = Field(description="Total faces detected")
    num_clusters: int = Field(description="Number of unique clusters identified")
    num_noise: int = Field(description="Number of unclustered faces")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    parameters: Dict = Field(description="Clustering parameters used")
    faces: List[FaceResult] = Field(description="Per-face clustering results")
    clusters: List[ClusterInfo] = Field(description="Per-cluster summary")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "face-clustering-api"
    version: str = "1.0.0"


class InfoResponse(BaseModel):
    """Service information"""
    service: str = "Face Clustering API"
    version: str = "1.0.0"
    description: str = "API for clustering faces in event photos using DBSCAN"
    endpoints: Dict[str, str] = {
        "POST /cluster": "Upload images and get face cluster assignments",
        "GET /health": "Health check",
        "GET /info": "Service information"
    }
    supported_models: List[str] = ["hog", "cnn"]
    clustering_algorithm: str = "DBSCAN"
    embedding_model: str = "dlib_face_recognition (128-d)"
