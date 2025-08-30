# Marksheet Extraction API

## Overview
This project provides an AI-powered API for extracting structured data from marksheets (images or PDFs). It uses Python, FastAPI, and OpenAI's Vision LLM for robust extraction, normalization, and confidence scoring.

## Features
- Accepts JPG, PNG, and PDF marksheets (≤10MB)
- Extracts candidate details, subject-wise marks, overall result, and issue info
- Returns structured JSON with confidence scores for each field
- Handles errors (invalid/large/wrong format files)
- Supports concurrent requests
- Modular, clean codebase

## Setup & Usage

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd marksheet-extracter
```

### 2. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up LLM credentials
- Add your OpenAI API key in `.env` (not committed to repo).

### 4. Run the API
```bash
uvicorn simple_main:app --reload
```

### 5. Test the API
- Use `/extract` endpoint with a marksheet file (image/PDF).
- See sample requests in `test/`.

## How to Run
1. Start the API server:
 ```bash
 uvicorn simple_main:app --reload
 ```
2. Send a POST request to `/extract` with a marksheet file.
3. Receive structured JSON output with confidence scores.

## Sample Mark Sheets
- See `test/` for example mark sheets used in testing.
