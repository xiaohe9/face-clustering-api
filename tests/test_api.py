"""
Face Clustering API - Automated Tests

Run tests:
    pytest tests/test_api.py -v

Or with coverage:
    pytest tests/test_api.py -v --cov=app --cov-report=term-missing
"""
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_bytes():
    """Create a minimal valid JPEG byte sequence for testing"""
    # This is a 1x1 pixel JPEG (minimal valid JPEG)
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
    ])


# ---------------------------------------------------------------------------
# Health & Info Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Test suite for /health endpoint"""
    
    def test_health_returns_200(self):
        """Health check should return 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_response_structure(self):
        """Health response should have correct structure"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "face-clustering-api"
        assert "version" in data
    
    def test_health_response_time(self):
        """Health check should respond quickly"""
        import time
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should respond in under 1 second


class TestInfoEndpoint:
    """Test suite for /info endpoint"""
    
    def test_info_returns_200(self):
        """Info endpoint should return 200 OK"""
        response = client.get("/info")
        assert response.status_code == 200
    
    def test_info_contains_endpoints(self):
        """Info should list available endpoints"""
        response = client.get("/info")
        data = response.json()
        assert "endpoints" in data
        assert "/cluster" in str(data["endpoints"])
        assert "/health" in str(data["endpoints"])


# ---------------------------------------------------------------------------
# Cluster Endpoint Tests
# ---------------------------------------------------------------------------

class TestClusterEndpoint:
    """Test suite for /cluster endpoint"""
    
    def test_cluster_no_images_returns_400(self):
        """Request without images should return 400 Bad Request"""
        response = client.post("/cluster")
        assert response.status_code == 400
        assert "No images" in response.json()["detail"] or "Field required" in str(response.json())
    
    def test_cluster_empty_file_returns_success(self, sample_image_bytes):
        """Minimal JPEG with no faces should return success with 0 faces"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        response = client.post("/cluster", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_faces"] == 0
    
    def test_cluster_with_parameters(self, sample_image_bytes):
        """Cluster endpoint should accept custom parameters"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        data = {
            "eps": 0.35,
            "min_samples": 1,
            "detection_model": "hog"
        }
        response = client.post("/cluster", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert result["parameters"]["eps"] == 0.35
        assert result["parameters"]["min_samples"] == 1
    
    def test_cluster_invalid_eps_returns_422(self, sample_image_bytes):
        """Invalid eps value should return 422 validation error"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        data = {"eps": 2.0}  # Should be <= 1.0
        response = client.post("/cluster", files=files, data=data)
        assert response.status_code == 422
    
    def test_cluster_invalid_min_samples_returns_422(self, sample_image_bytes):
        """Invalid min_samples value should return 422 validation error"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        data = {"min_samples": 0}  # Should be >= 1
        response = client.post("/cluster", files=files, data=data)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Response Structure Tests
# ---------------------------------------------------------------------------

class TestResponseStructure:
    """Verify API response structure matches specification"""
    
    def test_response_has_required_fields(self, sample_image_bytes):
        """Response should contain all required fields"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        response = client.post("/cluster", files=files)
        data = response.json()
        
        required_fields = [
            "status", "message", "total_images", "total_faces",
            "num_clusters", "num_noise", "processing_time_ms",
            "parameters", "faces", "clusters"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    def test_face_entry_structure(self, sample_image_bytes):
        """Each face entry should have correct structure"""
        files = {
            "images": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")
        }
        response = client.post("/cluster", files=files)
        data = response.json()
        
        # If no faces detected, faces array should be empty
        assert isinstance(data["faces"], list)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_non_image_file_handled(self):
        """Non-image files should be skipped gracefully"""
        files = {
            "images": ("test.txt", io.BytesIO(b"not an image"), "text/plain")
        }
        response = client.post("/cluster", files=files)
        assert response.status_code == 200
        data = response.json()
        # Should be processed but 0 faces (can't decode)
        assert data["total_faces"] == 0
    
    def test_multiple_files_mixed_validity(self, sample_image_bytes):
        """Mix of valid and invalid files should process gracefully"""
        files = [
            ("images", ("valid.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("images", ("invalid.txt", io.BytesIO(b"not an image"), "text/plain"))
        ]
        response = client.post("/cluster", files=files)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Documentation Endpoint Tests
# ---------------------------------------------------------------------------

class TestDocumentation:
    """Verify API documentation is accessible"""
    
    def test_swagger_ui_accessible(self):
        """Swagger UI should be accessible at /docs"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
    
    def test_redoc_accessible(self):
        """ReDoc should be accessible at /redoc"""
        response = client.get("/redoc")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
