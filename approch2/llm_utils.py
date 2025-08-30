import subprocess
import json
import re

def call_gemma_llm(raw_text: str):
    """
    Send OCR text to Gemma LLM for structured JSON output with all required fields.
    """
    prompt = f"""
You are a marksheet parser. Extract ALL student information from the following OCR text.

For candidate details, extract: Name, Father's Name, Mother's Name, Roll No, Registration No, Board, Institution, Exam Year.
For each subject, extract: name, full marks, marks obtained, and grade (if present).
For each field, include a confidence score between 0 and 1.
Return only valid JSON matching the schema below. Do NOT add any explanation, comments, or extra text. Do NOT mention the blurred face in the response.

{{
  "candidate_details": {{
    "name": "",
    "father_name": "",
    "mother_name": "",
    "roll_number": "",
    "registration_number": "",
    "exam_year": "",
    "board": "",
    "institution": ""
  }},
  "subjects": [
    {{"name": "", "max_marks": "", "marks_obtained": "", "grade": "", "confidence": ""}}
  ],
  "overall_result": {{
    "division": "",
    "total_marks": "",
    "percentage": "",
    "grade": "",
    "confidence": ""
  }},
  "issue_date": "",
  "issue_place": "",
  "confidence_overall": ""
}}

OCR Text:
{raw_text}
"""
    process = subprocess.run(
        ["ollama", "run", "gemma3:4b"],
        input=prompt,
        text=True,
        capture_output=True
    )

    response = process.stdout.strip()
    if not response:
        print("LLM error:", process.stderr)
        return {"error": "No response from LLM", "stderr": process.stderr}

    print("LLM response:", response)  

    def find_json(text):
        start = text.find('{')
        if start == -1:
            return None
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
        return None

    json_str = find_json(response)
    if json_str:
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            json_data = {"error": "Invalid JSON from LLM", "raw_output": response}
    else:
        json_data = {"error": "No JSON found in LLM output", "raw_output": response}

    return json_data