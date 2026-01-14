"""
FastAPI for Loan Quotation & Signature Extraction
Combines Gemini VLM extraction with YOLO signature detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import json
import os
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from extract_loan_gemini import LoanQuotationExtractorGemini
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Loan Quotation & Signature API", version="1.0")

# Fix PyTorch 2.6 weights_only issue
_original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = patched_load

# Load models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = r"C:\Users\soura\convolve4.0\signature\app\model\yolo_v8n_finetuned_hand_signatures.pt"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_extractor = LoanQuotationExtractorGemini(GEMINI_API_KEY)
yolo_model = YOLO(MODEL_PATH)

print("✅ Models loaded successfully")


class ImageInput(BaseModel):
    image_base64: str
    doc_id: str = "loan_001"
    page_no: int = 1
    signature_confidence: float = 0.2


class SignatureBBox(BaseModel):
    signature_id: int
    bbox: dict
    confidence: float


class LoanQuotationResponse(BaseModel):
    doc_id: str
    page_no: int
    fields: dict
    overall_confidence: float
    signatures: list[SignatureBBox]


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


def extract_signatures_yolo(image: Image.Image, confidence: float = 0.2) -> list:
    """Extract signatures using YOLO and return bounding boxes"""
    results = yolo_model.predict(image, conf=confidence, verbose=False)
    
    boxes_data = []
    predictions = results[0]
    
    for i, box in enumerate(predictions.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence_score = box.conf.item()
        
        bb_data = {
            "signature_id": i,
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2)
            },
            "confidence": round(confidence_score, 4)
        }
        boxes_data.append(bb_data)
    
    return boxes_data


@app.get("/")
def root():
    return {
        "service": "Loan Quotation & Signature Extraction API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "extract": "/extract (POST)"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gemini_api": "connected" if GEMINI_API_KEY else "missing",
        "yolo_model": "loaded"
    }


@app.post("/extract", response_model=LoanQuotationResponse)
def extract_loan_quotation(data: ImageInput):
    """
    Extract loan quotation fields and signature bounding boxes
    
    Input: Base64 encoded image
    Output: Loan quotation data + signature bounding boxes
    """
    try:
        # Decode image
        image = decode_base64_image(data.image_base64)
        
        # Save temporarily for processing
        temp_path = "temp_input.png"
        image.save(temp_path)
        
        # Extract loan quotation using Gemini
        loan_data = gemini_extractor.extract_from_image(
            temp_path,
            doc_id=data.doc_id,
            page_no=data.page_no
        )
        
        # Extract signatures using YOLO
        signature_bboxes = extract_signatures_yolo(image, data.signature_confidence)
        
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        # Combine results
        response = {
            "doc_id": loan_data["doc_id"],
            "page_no": loan_data["page_no"],
            "fields": loan_data["fields"],
            "overall_confidence": loan_data["overall_confidence"],
            "signatures": signature_bboxes
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract-loan-only")
def extract_loan_only(data: ImageInput):
    """Extract only loan quotation fields (no signature detection)"""
    try:
        image = decode_base64_image(data.image_base64)
        temp_path = "temp_input.png"
        image.save(temp_path)
        
        loan_data = gemini_extractor.extract_from_image(
            temp_path,
            doc_id=data.doc_id,
            page_no=data.page_no
        )
        
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        return loan_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract-signatures-only")
def extract_signatures_only(data: ImageInput):
    """Extract only signature bounding boxes (no loan data)"""
    try:
        image = decode_base64_image(data.image_base64)
        signature_bboxes = extract_signatures_yolo(image, data.signature_confidence)
        
        return {
            "doc_id": data.doc_id,
            "signatures": signature_bboxes,
            "count": len(signature_bboxes)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signature extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Loan Quotation & Signature Extraction API...")
    print("📄 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
