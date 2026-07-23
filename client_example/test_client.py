"""
Face Clustering API - Python Client Example

Usage:
    1. Start the API server first:
       docker run -p 8000:8000 face-clustering-api
    
    2. Run this client:
       python client_example/test_client.py --images /path/to/photos/
    
    3. Or test with single image:
       python client_example/test_client.py --image /path/to/photo.jpg
"""
import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

import requests


API_BASE_URL = "http://localhost:8000"


def check_health() -> bool:
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_BASE_URL}")
        print("  Make sure the Docker container is running:")
        print("  docker run -p 8000:8000 face-clustering-api")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def get_service_info():
    """Get service information"""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            print(f"\nService Info:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"✗ Failed to get info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")


def cluster_images(
    image_paths: List[str],
    eps: float = 0.4,
    min_samples: int = 2,
    detection_model: str = "hog"
) -> Optional[dict]:
    """
    Send images to the clustering API and return results.
    
    Args:
        image_paths: List of paths to image files
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples for cluster
        detection_model: 'hog' or 'cnn'
    
    Returns:
        API response JSON or None if failed
    """
    # Prepare multipart form data
    files = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print(f"⚠ File not found: {img_path}")
            continue
        files.append(("images", (path.name, open(path, "rb"), f"image/{path.suffix.lstrip('.')}")))
    
    if not files:
        print("✗ No valid images to process")
        return None
    
    # Prepare form parameters
    data = {
        "eps": eps,
        "min_samples": min_samples,
        "detection_model": detection_model
    }
    
    print(f"\n📤 Uploading {len(files)} image(s)...")
    print(f"   Parameters: eps={eps}, min_samples={min_samples}, model={detection_model}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/cluster",
            files=files,
            data=data,
            timeout=300  # 5 minutes for large batches
        )
        
        elapsed = time.time() - start_time
        
        # Close file handles
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Clustering complete in {elapsed:.1f}s (API: {result.get('processing_time_ms', 0):.0f}ms)")
            return result
        else:
            print(f"\n✗ API Error {response.status_code}:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(response.text)
            return None
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out. Try with fewer images or smaller files.")
        return None
    except Exception as e:
        print(f"\n✗ Request failed: {e}")
        return None


def print_results(result: dict):
    """Pretty print clustering results"""
    if not result or result.get("status") != "success":
        print("No results to display")
        return
    
    print("\n" + "=" * 60)
    print("  📊 CLUSTERING RESULTS")
    print("=" * 60)
    
    print(f"\n  Total Images Processed: {result['total_images']}")
    print(f"  Total Faces Detected:   {result['total_faces']}")
    print(f"  Unique People (Clusters): {result['num_clusters']}")
    print(f"  Unclustered Faces:      {result['num_noise']}")
    print(f"  Processing Time:        {result['processing_time_ms']}ms")
    
    print(f"\n  Parameters:")
    for key, value in result['parameters'].items():
        print(f"    {key}: {value}")
    
    if result['clusters']:
        print(f"\n  Clusters:")
        for cluster in result['clusters']:
            cid = cluster['cluster_id']
            cname = "Noise/Unclustered" if cid == -1 else f"Person {cid}"
            print(f"    {cname}: {cluster['face_count']} face(s)")
    
    if result['faces']:
        print(f"\n  Face Details (showing first 10):")
        for face in result['faces'][:10]:
            cid = face['cluster_id']
            cname = "UNCLUSTERED" if cid == -1 else f"Person_{cid}"
            bbox = face['bounding_box']
            print(f"    Face #{face['face_id']:2d} | {cname:12s} | "
                  f"{face['image_filename'][:20]:20s} | "
                  f"bbox({bbox['top']},{bbox['left']},{bbox['bottom']},{bbox['right']})")
        
        if len(result['faces']) > 10:
            print(f"    ... and {len(result['faces']) - 10} more faces")
    
    print("=" * 60)


def save_results(result: dict, output_path: str = "clustering_result.json"):
    """Save results to JSON file"""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Face Clustering API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  python test_client.py --health
  
  # Cluster single image
  python test_client.py --image photo.jpg
  
  # Cluster all images in a directory
  python test_client.py --images ./event_photos/
  
  # Custom clustering parameters
  python test_client.py --images ./photos/ --eps 0.35 --min-samples 3
        """
    )
    
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--info", action="store_true", help="Get service info")
    parser.add_argument("--image", type=str, help="Path to single image file")
    parser.add_argument("--images", type=str, help="Path to directory containing images")
    parser.add_argument("--eps", type=float, default=0.4, help="DBSCAN epsilon (default: 0.4)")
    parser.add_argument("--min-samples", type=int, default=2, help="Min samples per cluster (default: 2)")
    parser.add_argument("--model", type=str, default="hog", choices=["hog", "cnn"], help="Detection model")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--url", type=str, default=API_BASE_URL, help="API base URL")
    
    args = parser.parse_args()
    
    global API_BASE_URL
    API_BASE_URL = args.url.rstrip("/")
    
    # Health check
    if args.health:
        check_health()
        return
    
    # Service info
    if args.info:
        get_service_info()
        return
    
    # Check API is up before processing
    print("Checking API connection...")
    if not check_health():
        sys.exit(1)
    
    # Collect images
    image_paths = []
    
    if args.image:
        image_paths = [args.image]
    elif args.images:
        img_dir = Path(args.images)
        if not img_dir.is_dir():
            print(f"✗ Not a directory: {args.images}")
            sys.exit(1)
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_paths = [
            str(p) for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        
        if not image_paths:
            print(f"✗ No valid images found in {args.images}")
            sys.exit(1)
        
        print(f"Found {len(image_paths)} image(s) in {args.images}")
    else:
        parser.print_help()
        return
    
    # Call API
    result = cluster_images(
        image_paths=image_paths,
        eps=args.eps,
        min_samples=args.min_samples,
        detection_model=args.model
    )
    
    if result:
        print_results(result)
        
        if args.save:
            save_results(result, args.save)
        else:
            save_results(result)


if __name__ == "__main__":
    main()
