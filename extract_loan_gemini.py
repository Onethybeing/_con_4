"""Loan Quotation Extraction using Google Gemini Vision API"""

import os
import base64
import json
import requests
import re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a Document Intelligence model specialized in financial documents.

STRICT RULES:
- Extract ONLY values clearly visible in the document
- If a field is not present, return null
- Output MUST be valid JSON matching the schema exactly
- Preserve numbers exactly as they appear
- Handle multilingual text (English, Hindi, Gujarati)

Confidence scoring:
- Based on visibility and clarity
- Increase if field label is explicit
- overall_confidence is the minimum across all required fields"""


class LoanQuotationExtractorGemini:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        self.headers = {"Content-Type": "application/json"}
    
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def build_extraction_prompt(self) -> str:
        schema = """{
  "fields": {
    "dealer_name": {"value": "ABC Tractors Pvt Ltd", "confidence": 0.96},
    "model_name": {"value": "Mahindra 575 DI", "confidence": 0.98},
    "horse_power": {"value": 50, "confidence": 0.95},
    "asset_cost": {"value": 525000, "confidence": 0.97},
    "dealer_signature": {"present": true, "confidence": 0.93},
    "dealer_stamp": {"present": true, "confidence": 0.94}
  },
  "overall_confidence": 0.93
}"""
        return f"""{SYSTEM_PROMPT}

Extract from loan quotation:
1. dealer_name - Company/dealer name
2. model_name - Vehicle/asset model
3. horse_power - HP value
4. asset_cost - Total cost (numbers only)
5. dealer_signature - Is signature present?
6. dealer_stamp - Is stamp present?

Schema: {schema}

Return ONLY valid JSON, no markdown:"""
    
    def extract_from_image(self, image_path: str, doc_id: str = "loan_001", page_no: int = 1) -> Dict[str, Any]:
        print(f"🔍 Extracting: {Path(image_path).name}")
        
        base64_image = self.encode_image(image_path)
        mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", 
                     ".webp": "image/webp", ".gif": "image/gif"}.get(Path(image_path).suffix.lower(), "image/jpeg")
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": self.build_extraction_prompt()},
                    {"inline_data": {"mime_type": mime_type, "data": base64_image}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "topK": 32, "topP": 0.9, "maxOutputTokens": 4096}
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"✅ Extracted ({len(text)} chars)")
            
            return self._parse_json_response(text, doc_id, page_no)
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._get_empty_template(doc_id, page_no)
    
    def _parse_json_response(self, text: str, doc_id: str, page_no: int) -> Dict[str, Any]:
        text = re.sub(r'```json\n?', '', text).strip().strip('```')
        
        try:
            data = json.loads(text)
            data['doc_id'] = doc_id
            data['page_no'] = page_no
            return data
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    data['doc_id'] = doc_id
                    data['page_no'] = page_no
                    return data
                except:
                    pass
            
            template = self._get_empty_template(doc_id, page_no)
            template['raw_response'] = text[:500]
            return template
    
    def _get_empty_template(self, doc_id: str, page_no: int) -> Dict[str, Any]:
        return {
            "doc_id": doc_id,
            "page_no": page_no,
            "fields": {
                "dealer_name": {"value": None, "confidence": 0.0},
                "model_name": {"value": None, "confidence": 0.0},
                "horse_power": {"value": None, "confidence": 0.0},
                "asset_cost": {"value": None, "confidence": 0.0},
                "dealer_signature": {"present": False, "confidence": 0.0},
                "dealer_stamp": {"present": False, "confidence": 0.0}
            },
            "overall_confidence": 0.0
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {output_path}")


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable")
        print("Get key: https://makersuite.google.com/app/apikey")
        exit(1)
    
    extractor = LoanQuotationExtractorGemini(api_key)
    image_path = r"C:\Users\soura\convolve4.0\example_loan_quotation.webp"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    results = extractor.extract_from_image(image_path, "loan_quotation_001", 1)
    print(json.dumps(results, indent=2))
    
    extractor.save_results(results, r"C:\Users\soura\convolve4.0\loan_quotation_results.json")
    print("✅ Done!")
