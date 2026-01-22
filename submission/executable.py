#!/usr/bin/env python3
"""
Loan Quotation Document Extraction Pipeline
============================================
Hackathon Submission - Team Convolve_4

This pipeline extracts structured information from Indian loan quotation 
documents (tractor loans) including:
- Dealer Name (Hindi/English)
- Model Name
- Horse Power
- Asset Cost
- Signature Detection (bbox)
- Stamp Detection (bbox)

Usage:
    python executable.py <image_path> [--output result.json]
    python executable.py image.jpg -o result.json
    python executable.py folder/ --batch -o results.json

Requirements:
    - Python 3.11+
    - See requirements.txt for dependencies
"""

import argparse
import json
import sys
import time
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

# Import torch BEFORE paddleocr to avoid DLL loading issues on Windows
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_TO_PADDLEOCR = {
    "hindi": "hi", "english": "en", "gujarati": "gu", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "bengali": "bn", "marathi": "hi",
    "punjabi": "pa", "odia": "or", "malayalam": "ml", "assamese": "as",
    "devanagari": "hi", "latin": "en",
}

AVAILABLE_PADDLEOCR_LANGS = ["en", "hi", "ch", "ta", "te", "ka", "mr", "gu"]

# Model path - adjust based on your setup
MODEL_PATH = Path(__file__).parent / "models" / "sarvam-1" / "sarvam-1-q8_0.gguf"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a JSON extraction engine for Indian loan quotation documents.

STRICT RULES:
1. Output ONLY a valid JSON object with exactly 4 keys: dealer_name, model_name, horse_power, asset_cost
2. If a field is not found, use null
3. PRESERVE ORIGINAL LANGUAGE: Keep Hindi names in Hindi
4. horse_power must be a number (typically 20-100 HP) or null
5. asset_cost must be a number (remove commas, ₹, Rs) or null
6. No explanations - ONLY the JSON object"""

USER_PROMPT_TEMPLATE = """Extract 4 fields from this loan quotation OCR text.

FIELDS:
1. dealer_name - Shop/dealership name
2. model_name - Tractor model (e.g., POWERTRAC PT43GS)
3. horse_power - HP number (PT43 = 43 HP)
4. asset_cost - Price in rupees (6,35,000 = 635000)

EXAMPLE:
OCR: "बनकर पाटील ट्रैक्टर्स POWERTRAC PT43GS Price 6,35,000"
OUTPUT: {{"dealer_name": "बनकर पाटील ट्रैक्टर्स", "model_name": "POWERTRAC PT43GS", "horse_power": 43, "asset_cost": 635000}}

NOW EXTRACT FROM:
{ocr_text}

Return ONLY valid JSON:"""


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image: CLAHE + Deskew + Denoise."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Deskew using Hough lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# SCRIPT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_script(image: np.ndarray) -> str:
    """Detect dominant script using IndicPhotoOCR ViT classifier."""
    try:
        from IndicPhotoOCR.script_identification.vit.vit_infer import VIT_identifier
        identifier = VIT_identifier()
        
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        cv2.imwrite(temp_path, image)
        
        try:
            result = identifier.identify(temp_path, model_name='auto', device='cpu')
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        if result:
            script = str(result).lower().strip()
            script_map = {"marathi": "hindi", "devanagari": "hindi", "telegu": "telugu"}
            return script_map.get(script, script)
    except Exception:
        pass
    return "english"


# ═══════════════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════════════

def run_ocr(image: np.ndarray, lang: str = "hi") -> str:
    """Run PaddleOCR with specified language."""
    from paddleocr import PaddleOCR
    
    paddle_lang = SCRIPT_TO_PADDLEOCR.get(lang, lang)
    if paddle_lang not in AVAILABLE_PADDLEOCR_LANGS:
        paddle_lang = "hi"
    
    ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=False, show_log=False)
    result = ocr.ocr(image, cls=True)
    
    if not result or not result[0]:
        return ""
    
    lines = [line[1][0] for line in result[0] if line[1][1] > 0.3]
    return "\n".join(lines)


def run_multi_language_ocr(image: np.ndarray, primary_lang: str) -> str:
    """Run OCR with primary language + English, merge results."""
    texts = []
    text1 = run_ocr(image, primary_lang)
    if text1:
        texts.append(text1)
    
    if primary_lang not in ["en", "english", "latin"]:
        text2 = run_ocr(image, "en")
        if text2:
            texts.append(text2)
    
    all_lines = []
    seen = set()
    for text in texts:
        for line in text.split("\n"):
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                all_lines.append(line_clean)
                seen.add(line_clean)
    return "\n".join(all_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# STAMP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stamps(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect stamps using stamp2vec YOLO or fallback color detection."""
    stamps = []
    
    # Try stamp2vec first
    try:
        sys.path.insert(0, str(Path(__file__).parent / "stamp2vec"))
        from pipelines.detection.yolo_stamp import YoloStampPipeline
        from PIL import Image
        
        pipeline = YoloStampPipeline.from_pretrained("stamps-labs/yolo-stamp")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        boxes = pipeline(pil_image)
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4].tolist())
                stamps.append({"bbox": [x1, y1, x2, y2], "confidence": 0.85})
            return stamps
    except Exception:
        pass
    
    # Fallback: Color-based detection for blue/red circular regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    combined_mask = blue_mask | red_mask1 | red_mask2
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    min_area, max_area = (w * h) * 0.005, (w * h) * 0.15
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            if 0.5 < aspect_ratio < 2.0:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity > 0.3:
                    stamps.append({"bbox": [x, y, x + cw, y + ch], "confidence": min(0.7, circularity)})
    
    return stamps


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNATURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_signatures(image: np.ndarray, y_offset: int = 0) -> List[Dict[str, Any]]:
    """Detect signatures using ink color and stroke analysis."""
    signatures = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    blue_mask = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([130, 255, 200]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 100, 80]))
    ink_mask = blue_mask | black_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ink_mask = cv2.dilate(ink_mask, kernel, iterations=2)
    ink_mask = cv2.erode(ink_mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    min_area, max_area = (w * h) * 0.001, (w * h) * 0.20
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            if 1.2 < aspect_ratio < 10.0:
                signatures.append({
                    "bbox": [x, y + y_offset, x + cw, y + ch + y_offset],
                    "confidence": 0.6
                })
    return signatures


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_llm_extraction(ocr_text: str, model_path: str) -> Dict[str, Any]:
    """Extract fields using Sarvam-1 LLM."""
    from llama_cpp import Llama
    
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=6, verbose=False)
    
    user_prompt = USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    full_prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_prompt} [/INST]"
    
    output = llm(full_prompt, max_tokens=256, temperature=0.0, top_p=0.9,
                 stop=["</s>", "[INST]", "\n\n", "```"], echo=False)
    
    response_text = output["choices"][0]["text"].strip()
    
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            parsed = json.loads(response_text[json_start:json_end])
            result = {
                "dealer_name": parsed.get("dealer_name"),
                "model_name": parsed.get("model_name"),
                "horse_power": parsed.get("horse_power"),
                "asset_cost": parsed.get("asset_cost"),
            }
            # Validate
            if result["horse_power"]:
                try:
                    hp = int(result["horse_power"])
                    result["horse_power"] = hp if hp <= 200 else None
                except:
                    result["horse_power"] = None
            if result["asset_cost"]:
                try:
                    cost = int(str(result["asset_cost"]).replace(",", ""))
                    result["asset_cost"] = cost if cost >= 10000 else None
                except:
                    result["asset_cost"] = None
            return result
    except:
        pass
    return {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(image_path: str) -> Dict[str, Any]:
    """Process a single image through the full extraction pipeline."""
    start_time = time.time()
    doc_id = Path(image_path).stem
    
    # 1. Preprocessing
    image = preprocess_image(image_path)
    
    # 2. Script Detection
    script = detect_script(image)
    ocr_lang = SCRIPT_TO_PADDLEOCR.get(script, "hi")
    
    # 3. OCR
    ocr_text = run_multi_language_ocr(image, script)
    
    # 4. Stamp Detection (bottom half first)
    img_h, img_w = image.shape[:2]
    y_offset = img_h // 2
    bottom_half = image[y_offset:, :]
    
    stamps = detect_stamps(bottom_half)
    for stamp in stamps:
        x1, y1, x2, y2 = stamp["bbox"]
        stamp["bbox"] = [x1, y1 + y_offset, x2, y2 + y_offset]
    
    if not stamps:
        stamps = detect_stamps(image)
    
    stamp_result = {
        "present": len(stamps) > 0,
        "bbox": stamps[0]["bbox"] if stamps else None,
        "confidence": stamps[0]["confidence"] if stamps else None
    }
    
    # 5. Signature Detection (bottom half only)
    signatures = detect_signatures(bottom_half, y_offset)
    signature_result = {
        "present": len(signatures) > 0,
        "bbox": signatures[0]["bbox"] if signatures else None,
        "confidence": signatures[0]["confidence"] if signatures else None
    }
    
    # 6. LLM Extraction
    if MODEL_PATH.exists():
        llm_result = run_llm_extraction(ocr_text, str(MODEL_PATH))
    else:
        llm_result = {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}
    
    processing_time = time.time() - start_time
    
    # Calculate confidence
    confidences = []
    if llm_result.get("dealer_name"): confidences.append(0.8)
    if llm_result.get("model_name"): confidences.append(0.9)
    if llm_result.get("horse_power"): confidences.append(0.85)
    if llm_result.get("asset_cost"): confidences.append(0.9)
    if stamp_result["present"]: confidences.append(stamp_result["confidence"])
    if signature_result["present"]: confidences.append(signature_result["confidence"])
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "doc_id": doc_id,
        "fields": {
            "dealer_name": llm_result.get("dealer_name"),
            "model_name": llm_result.get("model_name"),
            "horse_power": llm_result.get("horse_power"),
            "asset_cost": llm_result.get("asset_cost"),
            "signature": signature_result,
            "stamp": stamp_result
        },
        "confidence": round(overall_confidence, 2),
        "processing_time_sec": round(processing_time, 2),
        "script_detected": script,
        "ocr_language": ocr_lang
    }


def process_batch(input_path: str) -> List[Dict[str, Any]]:
    """Process all images in a folder."""
    results = []
    folder = Path(input_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_file in folder.iterdir():
        if img_file.suffix.lower() in image_extensions:
            print(f"Processing: {img_file.name}")
            try:
                result = process_image(str(img_file))
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"doc_id": img_file.stem, "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract loan quotation fields from document images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python executable.py image.jpg
  python executable.py image.jpg -o result.json
  python executable.py images/ --batch -o results.json
        """
    )
    parser.add_argument("input", help="Image file or folder path")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--batch", action="store_true", help="Process all images in folder")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            return 1
        results = process_batch(str(input_path))
    else:
        if not input_path.exists():
            print(f"Error: Image not found: {input_path}")
            return 1
        results = process_image(str(input_path))
    
    output_json = json.dumps(results, indent=2, ensure_ascii=False)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Results saved to: {args.output}")
    else:
        print(output_json)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
