from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from ocr_utils import extract_text_from_image, extract_text_from_pdf
from llm_utils import call_gemma_llm

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Marksheet Extractor with Gemma LLM")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome! Use POST /extract_marksheet/ to upload your marksheet (PDF/Image)."}

@app.post("/extract_marksheet/")
async def extract_marksheet(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        
        if file.filename.lower().endswith(".pdf"):
            text, confidences, _ = extract_text_from_pdf(file_path)
        else:
            text, confidences, _ = extract_text_from_image(file_path)

       
        min_length = 30  
        #  keywords which has to be extracted
        expected_keywords = [
            "marks", "roll", "subject", "name", "total", "university", "board", "grade", "division",
            "registration", "college", "school", "examination", "result", "obtained", "father", "mother",
            "hindi", "english", "mathematics", "science", "social", "drawing", "history", "geography",
            "semester", "cgpa", "sgpa", "credit", "point", "group", "date", "place",
            "certificate", "cum", "high school", "intermediate", "controller", "verified", "issued",
            "bachelor", "arts", "university", "arunachal", "pradesh", "west bengal", "uttar pradesh"
        ]

        
        def has_keywords(text, keywords, min_count=3):
            found = 0
            text_lower = text.lower()
            for kw in keywords:
                if kw in text_lower:
                    found += 1
            return found >= min_count

        if not text or len(text.strip()) < min_length:
            os.remove(file_path)
            print("OCR TEXT:", text)  
            return JSONResponse(content={
                "error": "OCR failed or image is too unclear. Please upload a clearer marksheet image."
            })

        if not has_keywords(text, expected_keywords):
            os.remove(file_path)
            print("OCR TEXT:", text)  
            return JSONResponse(content={
                "error": "Uploaded image does not appear to be a valid marksheet. Please check your file."
            })

        # modal calling
        json_data = call_gemma_llm(text)

        # confidence evaluation
        json_data["confidence_overall"] = str(round(sum(confidences)/len(confidences), 2)) if confidences else ""

        
        os.remove(file_path)

        return JSONResponse(content=json_data)
    except Exception as e:
        return {"error": str(e)}
