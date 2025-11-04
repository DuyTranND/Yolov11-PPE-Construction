"""
Basic test script to verify the API is working correctly.
Run with: python test_api.py
"""

import requests
import io
from PIL import Image, ImageDraw


def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (640, 480), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Test Construction Site Image", fill=(255, 255, 0))
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()


def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False


def test_detect_endpoint():
    """Test the /detect/ endpoint."""
    print("Testing /detect/ endpoint...")
    try:
        # Create test image
        image_bytes = create_test_image()
        
        # Send request
        files = {"image": ("test.png", image_bytes, "image/png")}
        response = requests.post("http://127.0.0.1:8000/detect/", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of detections: {len(result['detections'])}")
            print("\nDetections:")
            for det in result['detections']:
                print(f"  - {det['label']}: {det['confidence']:.2f} at {det['bbox']}")
            print(f"\nProcessed image (Base64): {result['processed_image'][:50]}...")
            print("✅ Detection test passed!\n")
            return True
        else:
            print(f"❌ Detection test failed: {response.json()}\n")
            return False
    except Exception as e:
        print(f"❌ Detection test failed: {e}\n")
        return False


def test_classes_endpoint():
    """Test the /classes endpoint."""
    print("Testing /classes endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/classes")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Classes test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Classes test failed: {e}\n")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PPE Detection API Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_endpoint()))
    results.append(("Classes Endpoint", test_classes_endpoint()))
    results.append(("Detection Endpoint", test_detect_endpoint()))
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("=" * 60)
