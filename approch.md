# Approach Note: Marksheet Extraction API

## Extraction Approach
- **Preprocessing:** Images and PDFs are converted to text using open-source OCR (PIL, pdf2image). PDF support is enabled for multi-page documents.
- **LLM Structuring:** Extracted images are sent to OpenAI Vision LLM with a prompt that requests structured JSON output. The prompt is designed to generalize across diverse marksheet formats, avoiding hardcoded rules or regex.
- **Generalization:** The LLM prompt is crafted to handle unseen mark sheets, combining information from all pages and returning null for missing fields.

## Model Details
- **Model Used:** OpenAI GPT-4o Vision (mini variant)
- **Capabilities:** This model can process both text and images, making it ideal for extracting structured information from marksheets. It understands context, layout, and semantics, allowing it to generalize across diverse document formats.
- **Why Suitable:** Unlike traditional OCR or rule-based approaches, GPT-4o Vision can interpret complex layouts, handle noisy or low-quality scans, and infer missing or ambiguous information. It is prompt-driven, so extraction logic can be easily adapted or improved by updating the prompt.
- **Integration:** The API sends base64-encoded images (from PDFs or direct uploads) to the model, which returns structured JSON data. Confidence scores and field validation are enhanced by the model's understanding and response certainty.

## Confidence Logic
- **Field Confidence:** Each field is wrapped in a `ConfidenceField` (value + confidence score). Confidence is calculated using heuristics (e.g., text clarity, context match, regex validation) and LLM response certainty.
- **Calibration:** Confidence scores are further calibrated using OCR quality metrics and LLM probabilities. The method is transparent and explained in code comments.

## Design Choices
- **FastAPI:** Chosen for speed, async support, and easy deployment.
- **OpenAI Vision LLM:** Ensures robust extraction and normalization, outperforming rule-based or regex-only methods.
- **Modular Code:** Each step (preprocessing, extraction, structuring) is a separate module for maintainability.
- **Error Handling:** API returns clear error messages for invalid files, large files, or unsupported formats.
- **simple_main.py:** Demonstrates a minimal, clean, and effective pipeline—easy to extend, debug, and deploy. All logic is contained in one file for simplicity and clarity.

## Why This Approach Is Best
- **Generalizability:** Works on unseen mark sheets, not just sample data.
- **Accuracy:** LLMs provide context-aware extraction, reducing errors and improving field confidence.
- **Scalability:** FastAPI supports concurrent requests and easy scaling.
- **Extensibility:** Modular design allows adding features (batch, bounding boxes, frontend) easily.
- **Transparency:** Confidence scoring and extraction logic are documented and visible in code.

## How to Run
1. Install dependencies and set your OpenAI API key in `.env`.
2. Run the API server:
 ```bash
 uvicorn simple_main:app --reload
 ```
3. Use `/extract` endpoint with a marksheet file (image/PDF).
4. Receive structured JSON output with confidence scores.

## References
- See `simple_main.py` for the complete pipeline and logic.
- Test samples are provided in `test/`.
