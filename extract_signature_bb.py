"""
Extract hand signatures and save bounding box coordinates
Uses the cloned Hand-Signature-Extraction YOLO v8n model
"""

import json
import cv2
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Fix PyTorch 2.6 weights_only issue by patching torch.load
_original_load = torch.load
def patched_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)
torch.load = patched_load

# Load the fine-tuned YOLO model
MODEL_PATH = r"C:\Users\soura\convolve4.0\signature\app\model\yolo_v8n_finetuned_hand_signatures.pt"
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# Input image path
IMAGE_PATH = r"C:\Users\soura\convolve4.0\ChatGPT Image Jan 14, 2026, 12_01_50 AM.png"
OUTPUT_DIR = r"C:\Users\soura\convolve4.0"

def extract_signatures(image_path, confidence=0.2):
    """
    Extract signatures from image using YOLO v8n
    Returns: bounding boxes and results
    """
    print(f"🔍 Loading image: {Path(image_path).name}")
    
    # Read image
    image = Image.open(image_path)
    image_cv = cv2.imread(image_path)
    
    # Run YOLO inference
    print(f"🤖 Running YOLO v8n signature detection...")
    results = model.predict(image, conf=confidence)
    
    # Extract bounding boxes
    boxes_data = []
    predictions = results[0]
    
    print(f"✅ Found {len(predictions.boxes)} signatures")
    
    for i, box in enumerate(predictions.boxes):
        # Get coordinates
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
        print(f"  Signature {i}: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}) | Confidence: {confidence_score:.4f}")
    
    return boxes_data, predictions, image, image_cv

def save_results(boxes_data, predictions, image, image_cv, output_dir):
    """Save bounding box JSON and annotated image"""
    
    # Save bounding boxes to JSON
    json_path = Path(output_dir) / "signature_bb_coordinates.json"
    with open(json_path, 'w') as f:
        json.dump(boxes_data, f, indent=2)
    print(f"💾 Saved BB coordinates: {json_path}")
    
    # Draw bounding boxes on image
    image_with_bb = image.copy()
    draw = ImageDraw.Draw(image_with_bb)
    
    for i, box_data in enumerate(boxes_data):
        bbox = box_data["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw label
        label = f"Sig {i} ({box_data['confidence']:.2f})"
        draw.text((x1, y1 - 10), label, fill="red")
    
    # Save annotated image
    img_path = Path(output_dir) / "signature_with_bb.png"
    image_with_bb.save(img_path)
    print(f"🖼️  Saved annotated image: {img_path}")
    
    # Also save using YOLO's save functionality
    results_dir = Path(output_dir) / "signature_detections"
    results_dir.mkdir(exist_ok=True)
    for i, box in enumerate(predictions.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop = image.crop((x1, y1, x2, y2))
        crop_path = results_dir / f"signature_{i}.png"
        crop.save(crop_path)
    
    print(f"✂️  Saved {len(predictions.boxes)} cropped signatures to: {results_dir}")

if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        exit(1)
    
    print("="*60)
    print("🚀 Hand Signature Extraction & BB Detection")
    print("="*60)
    
    # Extract signatures
    boxes_data, predictions, image, image_cv = extract_signatures(IMAGE_PATH, confidence=0.2)
    
    # Save results
    save_results(boxes_data, predictions, image, image_cv, OUTPUT_DIR)
    
    print("="*60)
    print("✅ All done!")
    print("="*60)
