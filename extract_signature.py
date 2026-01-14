"""Signature Detection and Bounding Box Extraction"""

import sys
sys.path.append(r"C:\Users\soura\convolve4.0\signature\app")

import torch
# Patch torch.load to allow legacy models
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO
from PIL import Image, ImageDraw
import json
from pathlib import Path

# Load the model
model = YOLO(r"C:\Users\soura\convolve4.0\signature\app\model\yolo_v8n_finetuned_hand_signatures.pt")

# Input image
image_path = r"C:\Users\soura\convolve4.0\ChatGPT Image Jan 14, 2026, 12_01_50 AM.png"
image = Image.open(image_path)

print(f"🔍 Detecting signatures in: {Path(image_path).name}")

# Run prediction
results = model.predict(image, conf=0.2)

# Extract bounding boxes
detections = []
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(confidence, 3),
            "class": "signature"
        })

print(f"✅ Found {len(detections)} signature(s)")

# Save JSON
output_json = {
    "image": Path(image_path).name,
    "detections": detections,
    "total_signatures": len(detections)
}

json_path = r"C:\Users\soura\convolve4.0\signature_detection_results.json"
with open(json_path, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"💾 JSON saved: {json_path}")

# Draw bounding boxes on image
draw = ImageDraw.Draw(image)
for det in detections:
    bbox = det["bbox"]
    draw.rectangle(bbox, outline="red", width=3)
    draw.text((bbox[0], bbox[1]-20), f"Signature {det['confidence']:.2f}", fill="red")

# Save annotated image
output_image_path = r"C:\Users\soura\convolve4.0\signature_detected.png"
image.save(output_image_path)
print(f"🖼️  Annotated image saved: {output_image_path}")

# Display results
print("\n" + "="*60)
print("DETECTION RESULTS:")
print("="*60)
print(json.dumps(output_json, indent=2))
print("\n✅ Done!")
