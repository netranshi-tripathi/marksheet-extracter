import os
import json
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# PDF processing imports
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
    print("✅ PDF support enabled")
except ImportError:
    PDF_SUPPORT = False
    print("❌ PDF support disabled. Install: pip install pdf2image")
    print("   Also install poppler-utils for your OS")

# Load environment variables
load_dotenv()
OPENAI_API_KEY="sk-proj-KbF_8rcItc-UkYzI587yobp[oo-GjGoTxkVx7UCsYBBTQjQXuVvWDAwk3GQF9dCQ-PTqxPPMuYzIQnAGJr6BT3BlbkFJg9pI9oUyZMVy_V13mBekGvoyyMY6nsQWBhr7FbJJcSbn2hTldgPvgB4W9Vt_1kuItg44ay4ogA"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set OPENAI_API_KEY in your .env file")
    print("Example .env file content:")
    print("OPENAI_API_KEY=sk-proj-your_actual_api_key_here")
    exit(1)

app = FastAPI(
    title="Marksheet Extraction API",
    description="AI-powered marksheet data extraction API using OpenAI Vision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic Models
class ConfidenceField(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

class CandidateDetails(BaseModel):
    name: ConfidenceField
    father_mother_name: ConfidenceField
    roll_no: ConfidenceField
    registration_no: ConfidenceField
    date_of_birth: ConfidenceField
    exam_year: ConfidenceField
    board_university: ConfidenceField
    institution: ConfidenceField

class SubjectMark(BaseModel):
    subject: ConfidenceField
    max_marks_credits: ConfidenceField
    obtained_marks_credits: ConfidenceField
    grade: ConfidenceField

class IssueDetails(BaseModel):
    issue_date: ConfidenceField
    issue_place: ConfidenceField

class OverallResult(BaseModel):
    result_grade_division: ConfidenceField
    percentage: ConfidenceField
    cgpa: ConfidenceField

class MarksheetExtraction(BaseModel):
    candidate_details: CandidateDetails
    subjects: List[SubjectMark]
    overall_result: OverallResult
    issue_details: IssueDetails
    extraction_metadata: Dict[str, Any]

# Helper Functions
def calculate_confidence(value: str, field_type: str) -> float:
    """Calculate confidence based on field type"""
    if not value or not value.strip():
        return 0.0
    
    base_confidence = 0.8
    value = value.strip()
    
    if field_type == "name":
        if any(char.isdigit() for char in value):
            base_confidence -= 0.3
        if len(value.split()) >= 2:
            base_confidence += 0.1
    
    elif field_type == "number":
        if value.replace('.', '').replace('%', '').isdigit():
            base_confidence += 0.1
    
    elif field_type == "date":
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', value):
            base_confidence += 0.1
    
    return max(min(base_confidence, 1.0), 0.1)

def create_confidence_field(value: Any, field_type: str) -> ConfidenceField:
    """Create confidence field"""
    if value is None or value == "null":
        return ConfidenceField(value=None, confidence=0.0)
    
    value_str = str(value).strip()
    if not value_str:
        return ConfidenceField(value=None, confidence=0.0)
    
    confidence = calculate_confidence(value_str, field_type)
    
    return ConfidenceField(value=value_str, confidence=confidence)

def process_pdf_to_images(content: bytes) -> List[str]:
    """Convert PDF pages to base64 images"""
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=400, 
            detail="PDF processing not available. Please install pdf2image and poppler-utils"
        )
    
    try:
        print("📄 Converting PDF pages to images...")
        
        # Convert PDF to images (limit to first 5 pages for API efficiency)
        images = convert_from_bytes(
            content, 
            dpi=200,  # Good quality but not too large
            first_page=1,
            last_page=5,  # Limit pages to avoid timeout
            fmt='PNG'
        )
        
        print(f"📄 Converted {len(images)} pages from PDF")
        
        base64_images = []
        for i, image in enumerate(images):
            print(f"📄 Processing page {i+1}...")
            
            # Resize if too large
            max_size = 800
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=True, quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            base64_images.append(img_base64)
        
        return base64_images
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to process PDF: {str(e)}"
        )

def process_image(content: bytes) -> str:
    """Convert single image to base64"""
    try:
        image = Image.open(BytesIO(content))
        
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Reduce image size to avoid API limits
        max_size = 800
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Image resized to: {image.size}")
        
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file - supports images and PDFs"""
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Updated to include PDF support
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.pdf'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Check PDF support
    if file_extension == '.pdf' and not PDF_SUPPORT:
        raise HTTPException(
            status_code=400,
            detail="PDF processing not available. Install pdf2image and poppler-utils"
        )

def clean_response_text(text: str) -> str:
    """Clean OpenAI response"""
    if not text:
        return ""
    
    text = text.replace('``````', '')
    text = text.strip()
    
    if '{' in text and '}' in text:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
    
    return text

async def extract_with_openai_multi_page(images_b64: List[str]) -> Dict[str, Any]:
    """Extract data from multiple images (PDF pages) using OpenAI"""
    
    system_prompt = """You are an expert at extracting information from marksheets and academic documents.

You will receive multiple images that are pages from a single marksheet document. Analyze ALL pages together and extract comprehensive information.

Return the data in this exact JSON format:

{
    "candidate_details": {
        "name": "student name or null",
        "father_mother_name": "parent name or null", 
        "roll_no": "roll number or null",
        "registration_no": "registration number or null",
        "date_of_birth": "date of birth or null",
        "exam_year": "exam year or null",
        "board_university": "board/university name or null",
        "institution": "institution name or null"
    },
    "subjects": [
        {
            "subject": "subject name",
            "max_marks_credits": "max marks",
            "obtained_marks_credits": "obtained marks", 
            "grade": "grade"
        }
    ],
    "overall_result": {
        "result_grade_division": "overall result or null",
        "percentage": "percentage or null",
        "cgpa": "cgpa or null"
    },
    "issue_details": {
        "issue_date": "issue date or null",
        "issue_place": "issue place or null"
    }
}

IMPORTANT: 
1. Combine information from ALL pages
2. Look for subjects across all pages
3. Return ONLY valid JSON, no explanations
4. Use null for missing fields"""

    try:
        print(f"🤖 Sending {len(images_b64)} pages to OpenAI...")
        
        # Prepare content with all images
        content = [
            {
                "type": "text",
                "text": f"Extract all information from this {len(images_b64)}-page marksheet document. Analyze all pages together."
            }
        ]
        
        # Add all images to the content
        for i, img_b64 in enumerate(images_b64):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
            print(f"📄 Added page {i+1} to request")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=4000,
            temperature=1
        )
        
        if not response.choices or not response.choices[0].message.content:
            print("❌ Empty response from OpenAI")
            raise HTTPException(status_code=500, detail="Empty response from OpenAI")
        
        response_content = response.choices[0].message.content.strip()
        print(f"✅ Received response from OpenAI")
        
        # Clean and parse response
        clean_content = clean_response_text(response_content)
        
        if not clean_content:
            print("❌ No content after cleaning")
            raise HTTPException(status_code=500, detail="No valid content in response")
        
        try:
            parsed_data = json.loads(clean_content)
            print("✅ Successfully parsed JSON")
            return parsed_data
            
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON parsing failed: {json_error}")
            # Return fallback response
            return {
                "candidate_details": {
                    "name": None, "father_mother_name": None, "roll_no": None,
                    "registration_no": None, "date_of_birth": None, "exam_year": None,
                    "board_university": None, "institution": None
                },
                "subjects": [],
                "overall_result": {
                    "result_grade_division": None, "percentage": None, "cgpa": None
                },
                "issue_details": {
                    "issue_date": None, "issue_place": None
                }
            }
        
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def extract_with_openai(image_b64: str) -> Dict[str, Any]:
    """Extract data from single image using OpenAI"""
    return await extract_with_openai_multi_page([image_b64])

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    supported_formats = ["JPG", "JPEG", "PNG", "WebP"]
    if PDF_SUPPORT:
        supported_formats.append("PDF")
    
    return {
        "message": "Marksheet Extraction API",
        "status": "active",
        "version": "1.0.0",
        "supported_formats": supported_formats,
        "max_file_size": "10MB",
        "pdf_support": PDF_SUPPORT,
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    supported_formats = ["JPG", "JPEG", "PNG", "WebP"]
    if PDF_SUPPORT:
        supported_formats.append("PDF")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "pdf_support": PDF_SUPPORT,
        "supported_formats": supported_formats
    }

@app.post("/extract", response_model=MarksheetExtraction)
async def extract_marksheet(file: UploadFile = File(...)):
    """Extract information from marksheet image or PDF"""
    
    print(f"📄 Processing file: {file.filename}")
    
    try:
        validate_file(file)
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Process based on file type
        if file_extension == '.pdf':
            print("📄 Processing PDF file...")
            images_b64 = process_pdf_to_images(content)
            print(f"📄 Got {len(images_b64)} pages from PDF")
            raw_data = await extract_with_openai_multi_page(images_b64)
            pages_processed = len(images_b64)
        else:
            print("🖼️ Processing image file...")
            image_b64 = process_image(content)
            raw_data = await extract_with_openai(image_b64)
            pages_processed = 1
        
        print("📊 Processing results...")
        
        # Process candidate details
        candidate_raw = raw_data.get("candidate_details", {})
        candidate_details = CandidateDetails(
            name=create_confidence_field(candidate_raw.get("name"), "name"),
            father_mother_name=create_confidence_field(candidate_raw.get("father_mother_name"), "name"),
            roll_no=create_confidence_field(candidate_raw.get("roll_no"), "number"),
            registration_no=create_confidence_field(candidate_raw.get("registration_no"), "number"),
            date_of_birth=create_confidence_field(candidate_raw.get("date_of_birth"), "date"),
            exam_year=create_confidence_field(candidate_raw.get("exam_year"), "number"),
            board_university=create_confidence_field(candidate_raw.get("board_university"), "general"),
            institution=create_confidence_field(candidate_raw.get("institution"), "general")
        )
        
        # Process subjects
        subjects_raw = raw_data.get("subjects", [])
        subjects = []
        for subject_data in subjects_raw:
            if isinstance(subject_data, dict):
                subject = SubjectMark(
                    subject=create_confidence_field(subject_data.get("subject"), "general"),
                    max_marks_credits=create_confidence_field(subject_data.get("max_marks_credits"), "number"),
                    obtained_marks_credits=create_confidence_field(subject_data.get("obtained_marks_credits"), "number"),
                    grade=create_confidence_field(subject_data.get("grade"), "grade")
                )
                subjects.append(subject)
        
        print(f"📚 Found {len(subjects)} subjects")
        
        # Process overall result
        result_raw = raw_data.get("overall_result", {})
        overall_result = OverallResult(
            result_grade_division=create_confidence_field(result_raw.get("result_grade_division"), "grade"),
            percentage=create_confidence_field(result_raw.get("percentage"), "number"),
            cgpa=create_confidence_field(result_raw.get("cgpa"), "number")
        )
        
        # Process issue details
        issue_raw = raw_data.get("issue_details", {})
        issue_details = IssueDetails(
            issue_date=create_confidence_field(issue_raw.get("issue_date"), "date"),
            issue_place=create_confidence_field(issue_raw.get("issue_place"), "general")
        )
        
        # Enhanced metadata
        metadata = {
            "filename": file.filename,
            "file_type": file_extension,
            "file_size_mb": round(len(content) / (1024 * 1024), 2),
            "extraction_method": "OpenAI GPT-4o-mini Vision",
            "pages_processed": pages_processed,
            "timestamp": datetime.now().isoformat(),
            "total_subjects": len(subjects),
            "status": "success"
        }
        
        print("✅ Extraction completed successfully!")
        
        return MarksheetExtraction(
            candidate_details=candidate_details,
            subjects=subjects,
            overall_result=overall_result,
            issue_details=issue_details,
            extraction_metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Marksheet Extraction API...")
    
    supported_formats = ["JPG", "JPEG", "PNG", "WebP"]
    if PDF_SUPPORT:
        supported_formats.append("PDF")
    else:
        print("⚠️  PDF support not available")
        print("   Install with: pip install pdf2image")
        print("   Also install poppler-utils for your OS")
    
    print(f"📁 Supported: {', '.join(supported_formats)}")
    print("📊 Max size: 10MB")
    print("🤖 Model: OpenAI GPT-4o-mini")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔍 Health: http://localhost:8000/health")
    
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
