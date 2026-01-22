# 🚜 Loan Quotation Document Extraction Pipeline

> **Hackathon Submission - Team Convolve_4**

An end-to-end pipeline for extracting structured information from Indian loan quotation documents (specifically tractor loan quotations) using OCR, script detection, and LLM-based field extraction.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Flow](#pipeline-flow)
- [Components](#components)
- [Cost Analysis](#cost-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Output Schema](#output-schema)
- [Performance Metrics](#performance-metrics)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOAN QUOTATION EXTRACTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  INPUT   │───▶│ PREPROC  │───▶│  SCRIPT  │───▶│   OCR    │              │
│  │  IMAGE   │    │  CLAHE   │    │   DET    │    │ PaddleOCR│              │
│  │  (JPG)   │    │ Deskew   │    │ViT CNN   │    │ Hi/En/Gu │              │
│  └──────────┘    │ Denoise  │    │12-class  │    └────┬─────┘              │
│                  └──────────┘    └──────────┘         │                     │
│                                                        │                     │
│  ┌─────────────────────────────────────────────────────┼────────────────┐   │
│  │                      PARALLEL PROCESSING            │                │   │
│  │  ┌──────────────┐    ┌──────────────┐              │                │   │
│  │  │    STAMP     │    │  SIGNATURE   │              ▼                │   │
│  │  │  DETECTION   │    │  DETECTION   │    ┌──────────────┐           │   │
│  │  │  stamp2vec   │    │  Color/Ink   │    │     LLM      │           │   │
│  │  │  YOLO-based  │    │  Analysis    │    │  Sarvam-1    │           │   │
│  │  │  + Fallback  │    │  Contours    │    │  2B Q8_0     │           │   │
│  │  └──────┬───────┘    └──────┬───────┘    │  Few-shot    │           │   │
│  │         │                   │            └──────┬───────┘           │   │
│  └─────────┼───────────────────┼───────────────────┼───────────────────┘   │
│            │                   │                   │                        │
│            ▼                   ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        RESULT AGGREGATION                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  {                                                          │    │   │
│  │  │    "doc_id": "invoice_001",                                 │    │   │
│  │  │    "fields": {                                              │    │   │
│  │  │      "dealer_name": "बनकर पाटील ट्रैक्टर्स",                   │    │   │
│  │  │      "model_name": "POWERTRAC PT43GS",                      │    │   │
│  │  │      "horse_power": 43,                                     │    │   │
│  │  │      "asset_cost": 635000,                                  │    │   │
│  │  │      "signature": {"present": true, "bbox": [...]},         │    │   │
│  │  │      "stamp": {"present": true, "bbox": [...]}              │    │   │
│  │  │    },                                                       │    │   │
│  │  │    "confidence": 0.85                                       │    │   │
│  │  │  }                                                          │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE STAGES                          │
└─────────────────────────────────────────────────────────────────┘

     ┌───────┐
     │ START │
     └───┬───┘
         │
         ▼
┌─────────────────┐
│ 1. PREPROCESS   │ ◄── CLAHE contrast enhancement
│    IMAGE        │ ◄── Hough line deskew
│                 │ ◄── Non-local means denoising
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. SCRIPT       │ ◄── IndicPhotoOCR ViT classifier
│    DETECTION    │ ◄── 12-class: hi/en/gu/ta/te/kn/bn/mr/pa/or/ml/as
│                 │ ◄── Confidence threshold: 0.90
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. OCR          │ ◄── PaddleOCR v2.9.1
│    EXTRACTION   │ ◄── Primary language + English merge
│                 │ ◄── Confidence filter: >0.3
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 4. STAMP     │  │ 5. SIGNATURE │  │ 6. LLM       │
│ DETECTION    │  │ DETECTION    │  │ EXTRACTION   │
│              │  │              │  │              │
│ stamp2vec    │  │ HSV color    │  │ Sarvam-1 2B  │
│ YOLO model   │  │ thresholding │  │ GGUF Q8_0    │
│ + Fallback   │  │ Contour      │  │ Few-shot     │
│ color detect │  │ analysis     │  │ prompting    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │ 7. AGGREGATE    │
              │    RESULTS      │
              │    + Confidence │
              └────────┬────────┘
                       │
                       ▼
                  ┌─────────┐
                  │ OUTPUT  │ ──► JSON
                  │  JSON   │
                  └─────────┘
```

---

## 🧩 Components

### 1. Image Preprocessing
| Technique | Purpose | Library |
|-----------|---------|---------|
| CLAHE | Contrast enhancement | OpenCV |
| Hough Transform | Deskew correction | OpenCV |
| Non-local Means | Noise reduction | OpenCV |

### 2. Script Detection
| Model | Classes | Accuracy |
|-------|---------|----------|
| IndicPhotoOCR ViT | 12 Indian scripts | ~90% |

Supported scripts: Hindi, English, Gujarati, Tamil, Telugu, Kannada, Bengali, Marathi, Punjabi, Odia, Malayalam, Assamese

### 3. OCR Engine
| Engine | Languages | Features |
|--------|-----------|----------|
| PaddleOCR 2.9.1 | 80+ | Angle classification, multi-language |

### 4. Stamp Detection
| Method | Model | Source |
|--------|-------|--------|
| Primary | YOLO-Stamp | stamp2vec (stamps-labs) |
| Fallback | HSV color + contour | Custom |

### 5. Signature Detection
| Method | Features |
|--------|----------|
| Color thresholding | Blue/black ink detection |
| Contour analysis | Aspect ratio filtering (1.2-10.0) |
| Region restriction | Bottom half of document |

### 6. LLM Extraction
| Model | Size | Quantization | Context |
|-------|------|--------------|---------|
| Sarvam-1 | 2B | Q8_0 GGUF | 2048 tokens |

---

## 💰 Cost Analysis

### Computational Costs (per document)

| Component | CPU Time | Memory | GPU Required |
|-----------|----------|--------|--------------|
| Preprocessing | ~0.5s | 100MB | No |
| Script Detection | ~2s | 500MB | No (CPU) |
| OCR | ~5s | 800MB | No (CPU) |
| Stamp Detection | ~3s | 400MB | No (CPU) |
| Signature Detection | ~0.2s | 50MB | No |
| LLM Extraction | ~20s | 2GB | No (CPU) |
| **Total** | **~30s** | **~3GB peak** | **No** |

### API Cost Comparison

| Approach | Cost per 1000 docs | Latency |
|----------|-------------------|---------|
| **Our Pipeline (Local)** | **$0** | ~30s/doc |
| GPT-4 Vision | ~$30 | ~5s/doc |
| Azure Document Intelligence | ~$15 | ~3s/doc |
| Google Document AI | ~$10 | ~2s/doc |

### Hardware Requirements

| Tier | RAM | CPU | Processing Time |
|------|-----|-----|-----------------|
| Minimum | 8GB | 4 cores | ~60s/doc |
| Recommended | 16GB | 8 cores | ~30s/doc |
| Optimal | 32GB | 12+ cores | ~15s/doc |

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Onethybeing/_con_4.git
cd _con_4

# Create conda environment
conda create -n convolve python=3.11 -y
conda activate convolve

# Install dependencies
pip install -r requirements.txt

# Download models (if not included)
# - Sarvam-1 Q8_0 GGUF (~1.7GB)
# - IndicPhotoOCR ViT models (auto-download)
# - stamp2vec YOLO weights (auto-download from HuggingFace)
```

---

## 📖 Usage

### Single Image
```bash
python executable.py image.jpg -o result.json
```

### Batch Processing
```bash
python executable.py images_folder/ --batch -o results.json
```

### Python API
```python
from executable import process_image

result = process_image("quotation.jpg")
print(result["fields"]["dealer_name"])
print(result["fields"]["asset_cost"])
```

---

## 📄 Output Schema

```json
{
  "doc_id": "90ae06be-dcab-44a3-a0f8-11dbe499d34f",
  "fields": {
    "dealer_name": "बनकर पाटील ट्रैक्टर्स",
    "model_name": "POWERTRAC TRACTOR PT43GS",
    "horse_power": 43,
    "asset_cost": 635000,
    "signature": {
      "present": true,
      "bbox": [26, 1496, 234, 1524],
      "confidence": 0.6
    },
    "stamp": {
      "present": true,
      "bbox": [756, 247, 1008, 524],
      "confidence": 0.85
    }
  },
  "confidence": 0.82,
  "processing_time_sec": 45.2,
  "script_detected": "hindi",
  "ocr_language": "hi"
}
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Field Extraction Accuracy | ~85% |
| Stamp Detection Recall | ~90% |
| Signature Detection Recall | ~75% |
| Average Processing Time | 30-60s |
| Supported Scripts | 12 Indian languages |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.x |
| OCR | PaddleOCR 2.9.1 |
| LLM | llama.cpp + Sarvam-1 |
| Script ID | IndicPhotoOCR (ViT) |
| Object Detection | stamp2vec (YOLO) |
| Image Processing | OpenCV, NumPy |

---

## 📁 Project Structure

```
submission/
├── executable.py        # Main extraction pipeline
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── utils/              # Supporting modules
│   └── __init__.py
└── sample_output/
    └── result.json     # Example output
```

---

## 👥 Team Convolve_4

Built with ❤️ for the Hackathon

---

## 📜 License

MIT License
