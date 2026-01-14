"""
Test client for Loan Quotation & Signature API
Demonstrates how to call the API with base64 image
"""

import base64
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_full_extraction(image_path: str):
    """Test complete extraction (loan data + signatures)"""
    print("="*60)
    print("🧪 Testing Full Extraction (Loan + Signatures)")
    print("="*60)
    
    # Encode image
    image_base64 = encode_image_to_base64(image_path)
    
    # Prepare request
    payload = {
        "image_base64": image_base64,
        "doc_id": "test_loan_001",
        "page_no": 1,
        "signature_confidence": 0.2
    }
    
    # Call API
    print(f"📤 Sending request to {API_URL}/extract")
    response = requests.post(f"{API_URL}/extract", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print("\n📊 RESULTS:")
        print(json.dumps(result, indent=2))
        
        # Save to file
        output_file = "api_extraction_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {output_file}")
        
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def test_loan_only(image_path: str):
    """Test loan extraction only"""
    print("\n" + "="*60)
    print("🧪 Testing Loan Extraction Only")
    print("="*60)
    
    image_base64 = encode_image_to_base64(image_path)
    
    payload = {
        "image_base64": image_base64,
        "doc_id": "test_loan_002",
        "page_no": 1
    }
    
    print(f"📤 Sending request to {API_URL}/extract-loan-only")
    response = requests.post(f"{API_URL}/extract-loan-only", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print("\n📄 Loan Data:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def test_signatures_only(image_path: str):
    """Test signature extraction only"""
    print("\n" + "="*60)
    print("🧪 Testing Signature Extraction Only")
    print("="*60)
    
    image_base64 = encode_image_to_base64(image_path)
    
    payload = {
        "image_base64": image_base64,
        "doc_id": "test_sig_001",
        "signature_confidence": 0.2
    }
    
    print(f"📤 Sending request to {API_URL}/extract-signatures-only")
    response = requests.post(f"{API_URL}/extract-signatures-only", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"\n✍️ Found {result['count']} signatures:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("🧪 Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print("✅ API is healthy")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health check failed: {response.status_code}")


if __name__ == "__main__":
    # Test image path
    IMAGE_PATH = r"C:\Users\soura\convolve4.0\ChatGPT Image Jan 14, 2026, 12_01_50 AM.png"
    
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        exit(1)
    
    print("🚀 Loan Quotation & Signature API Test Client")
    print(f"📷 Testing with: {Path(IMAGE_PATH).name}\n")
    
    # Check if API is running
    try:
        requests.get(API_URL, timeout=2)
    except:
        print("❌ API is not running!")
        print("Start the API first with: python api_loan_extraction.py")
        exit(1)
    
    # Run tests
    test_health()
    test_full_extraction(IMAGE_PATH)
    # test_loan_only(IMAGE_PATH)
    # test_signatures_only(IMAGE_PATH)
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
