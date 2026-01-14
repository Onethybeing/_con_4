# Loan Quotation & Signature Extraction System

AI-powered document intelligence system for extracting loan quotation information and detecting signatures using Google Gemini Vision and YOLO v8.

## 🎯 Features

- **Loan Quotation Extraction** - Extracts structured data from loan quotations using Gemini 2.5 Flash
  - Dealer name
  - Model name
  - Horse power
  - Asset cost
  - Signature presence detection
  - Stamp presence detection

- **Signature Detection** - Precise signature bounding box detection using fine-tuned YOLO v8n
  - Bounding box coordinates (x1, y1, x2, y2)
  - Confidence scores
  - Automatic signature cropping

- **FastAPI REST API** - Production-ready API for integration
  - Base64 image input
  - JSON structured output
  - Multiple endpoints for different use cases

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Onethybeing/_con_4.git
cd _con_4
```

### 2. Setup Environment

```bash
# Create conda environment
conda create -n myenv python=3.11
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

Get your free Gemini API key: https://makersuite.google.com/app/apikey

### 4. Run the API

```bash
python api_loan_extraction.py
```

API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## 📡 API Endpoints

### `POST /extract`
Extract both loan quotation data and signature bounding boxes.

**Request:**
```json
{
  "image_base64": "base64_encoded_image_string",
  "doc_id": "loan_001",
  "page_no": 1,
  "signature_confidence": 0.2
}
```

**Response:**
```json
{
  "doc_id": "loan_001",
  "page_no": 1,
  "fields": {
    "dealer_name": {"value": "ABC Tractors", "confidence": 0.98},
    "asset_cost": {"value": 525000, "confidence": 0.97},
    "dealer_signature": {"present": true, "confidence": 0.95}
  },
  "overall_confidence": 0.95,
  "signatures": [
    {
      "signature_id": 0,
      "bbox": {"x1": 820, "y1": 943, "x2": 977, "y2": 1066},
      "confidence": 0.79
    }
  ]
}
```

### `POST /extract-loan-only`
Extract only loan quotation fields (no signature detection).

### `POST /extract-signatures-only`
Extract only signature bounding boxes (no loan data).

### `GET /health`
Check API health status.

## 🧪 Testing

```bash
# Test with example image
python test_api_client.py
```

## 📁 Project Structure

```
_con_4/
├── api_loan_extraction.py      # FastAPI server
├── extract_loan_gemini.py      # Gemini extraction logic
├── extract_signature_bb.py     # YOLO signature detection
├── test_api_client.py          # API test client
├── signature/                  # YOLO model files
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── requirements.txt            # Python dependencies
```

## 🔧 Requirements

- Python 3.11+
- Google Gemini API key (free tier available)
- YOLO v8n model (included in signature/ folder)

## 📦 Dependencies

```
fastapi
uvicorn
pydantic
ultralytics
torch
pillow
opencv-python
requests
python-dotenv
```

## 🛠️ Development

### Run Standalone Scripts

**Loan Quotation Extraction:**
```bash
python extract_loan_gemini.py
```

**Signature Detection:**
```bash
python extract_signature_bb.py
```

## 🔐 Security

- **Never commit `.env` file** - Contains your API keys
- Use `.env.example` as template
- API keys are loaded from environment variables
- Add sensitive files to `.gitignore`

## 📊 Model Information

### Gemini 2.5 Flash
- Provider: Google AI
- Type: Vision Language Model
- Use: Document understanding and extraction
- Free tier: 60 requests/minute

### YOLO v8n
- Base: Ultralytics YOLO v8 nano
- Fine-tuned on: Hand signature datasets
- Task: Object detection (signatures)
- Speed: ~50ms inference

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Google Gemini API](https://ai.google.dev/)
- [Hand-Signature-Extraction](https://github.com/Thunderhead-exe/Hand-Signature-Extraction)

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ using AI
