"""
Face Clustering Engine - Refactored from fae_reg.py for API usage

Core pipeline:
1. Detect faces in images and extract 128-d embeddings
2. Cluster embeddings using DBSCAN
3. Return structured results
"""
import os
import io
import tempfile
import time
import cv2
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)

# Valid image extensions for processing
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')


def is_valid_image(filename: str) -> bool:
    """Check if file has a valid image extension"""
    return filename.lower().endswith(VALID_EXTENSIONS)


def load_image_from_bytes(file_bytes: bytes, filename: str = "unknown") -> Optional[np.ndarray]:
    """
        Load an image from bytes (uploaded file) into RGB numpy array.
        
    Args:
        file_bytes: Raw file bytes
        filename: Original filename (for logging)
        
    Returns:
        RGB image array or None if loading fails
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        # Decode image (OpenCV loads as BGR by default)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning(f"Failed to decode image: {filename}")
            return None
        # Convert BGR to RGB (face_recognition requires RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        logger.error(f"Error loading image {filename}: {e}")
        return None


def detect_and_encode(
    image_rgb: np.ndarray,
    filename: str,
    detection_model: str = "hog"
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Detect faces in a single image and extract 128-d embeddings.
    
    Args:
        image_rgb: RGB image array
        filename: Source image filename (for metadata)
        detection_model: "hog" (CPU) or "cnn" (GPU)
        
    Returns:
        Tuple of (embeddings_list, metadata_list)
    """
    embeddings = []
    metadata = []
    
    try:
        # Detect face locations: (top, right, bottom, left)
        face_locations = face_recognition.face_locations(image_rgb, model=detection_model)
        
        if not face_locations:
            logger.info(f"No faces detected in {filename}")
            return embeddings, metadata
        
        # Extract 128-dimensional face encodings
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        
        for box, encoding in zip(face_locations, face_encodings):
            embeddings.append(encoding)
            metadata.append({
                "image_filename": filename,
                "bounding_box": {
                    "top": box[0],
                    "right": box[1],
                    "bottom": box[2],
                    "left": box[3]
                }
            })
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    
    return embeddings, metadata


def cluster_embeddings(
    embeddings: np.ndarray,
    eps: float = 0.4,
    min_samples: int = 2
) -> np.ndarray:
    """
    Cluster face embeddings using DBSCAN.
    
    Args:
        embeddings: Array of 128-d face vectors (N x 128)
        eps: Maximum distance between two samples for neighborhood
        min_samples: Minimum points to form a core sample
        
    Returns:
        Cluster labels array (-1 indicates noise)
    """
    if len(embeddings) == 0:
        return np.array([])
    
    # DBSCAN with euclidean distance on L2-normalized face vectors
    clt = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1  # Use all CPU cores
    )
    clt.fit(embeddings)
    return clt.labels_


def build_clustering_results(
    all_metadata: List[Dict],
    labels: np.ndarray,
    total_images: int,
    processing_time_ms: float,
    parameters: Dict
) -> Dict:
    """
    Build structured API response from clustering results.
    
    Args:
        all_metadata: List of face metadata dicts
        labels: Cluster labels from DBSCAN
        total_images: Number of input images
        processing_time_ms: Processing duration
        parameters: Clustering parameters used
        
    Returns:
        Structured response dictionary matching ClusteringResponse model
    """
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = int(np.sum(labels == -1))
    
    # Build per-face results
    faces = []
    for face_id, (meta, label) in enumerate(zip(all_metadata, labels)):
        faces.append({
            "face_id": face_id,
            "cluster_id": int(label),
            "image_filename": meta["image_filename"],
            "bounding_box": meta["bounding_box"]
        })
    
    # Build per-cluster summary
    cluster_dict: Dict[int, List[int]] = {}
    for face in faces:
        cid = face["cluster_id"]
        if cid not in cluster_dict:
            cluster_dict[cid] = []
        cluster_dict[cid].append(face["face_id"])
    
    clusters = []
    for cid in sorted(cluster_dict.keys()):
        clusters.append({
            "cluster_id": cid,
            "face_count": len(cluster_dict[cid]),
            "face_ids": cluster_dict[cid]
        })
    
    return {
        "status": "success",
        "message": f"Clustering complete. Found {num_clusters} unique people from {len(faces)} faces.",
        "total_images": total_images,
        "total_faces": len(faces),
        "num_clusters": num_clusters,
        "num_noise": num_noise,
        "processing_time_ms": round(processing_time_ms, 2),
        "parameters": parameters,
        "faces": faces,
        "clusters": clusters
    }


def process_images(
    uploaded_files: List[Tuple[str, bytes]],
    eps: float = 0.4,
    min_samples: int = 2,
    detection_model: str = "hog"
) -> Dict:
    """
    Main entry point: Process uploaded images through face clustering pipeline.
    
    Args:
        uploaded_files: List of (filename, file_bytes) tuples
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        detection_model: Face detection model ("hog" or "cnn")
        
    Returns:
        Structured clustering results dictionary
    """
    start_time = time.time()
    
    all_embeddings = []
    all_metadata = []
    processed_images = 0
    skipped_images = 0
    
    # Step 1: Detect faces and extract embeddings from all images
    for filename, file_bytes in uploaded_files:
        if not is_valid_image(filename):
            logger.warning(f"Skipping non-image file: {filename}")
            skipped_images += 1
            continue
        
        image_rgb = load_image_from_bytes(file_bytes, filename)
        if image_rgb is None:
            skipped_images += 1
            continue
        
        embeddings, metadata = detect_and_encode(image_rgb, filename, detection_model)
        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)
        processed_images += 1
    
    if len(all_embeddings) == 0:
        processing_time_ms = (time.time() - start_time) * 1000
        return {
            "status": "success",
            "message": "No faces detected in any of the uploaded images.",
            "total_images": processed_images,
            "total_faces": 0,
            "num_clusters": 0,
            "num_noise": 0,
            "processing_time_ms": round(processing_time_ms, 2),
            "parameters": {
                "eps": eps,
                "min_samples": min_samples,
                "detection_model": detection_model
            },
            "faces": [],
            "clusters": []
        }
    
    # Step 2: Convert to numpy array and cluster
    embeddings_array = np.array(all_embeddings)
    labels = cluster_embeddings(embeddings_array, eps=eps, min_samples=min_samples)
    
    # Step 3: Build results
    processing_time_ms = (time.time() - start_time) * 1000
    parameters = {
        "eps": eps,
        "min_samples": min_samples,
        "detection_model": detection_model
    }
    
    return build_clustering_results(
        all_metadata, labels, processed_images, processing_time_ms, parameters
    )
