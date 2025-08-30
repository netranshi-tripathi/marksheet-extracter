import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path: str):
    """
    Extract text and confidence from image using pytesseract.
    Returns cleaned text (with line breaks) and list of (word, confidence) tuples.
    """
    image = Image.open(image_path)
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    finally:
        image.close()  

    lines = {}
    word_confidences = []

    for i, word in enumerate(data['text']):
        if word.strip():
            line_num = data['line_num'][i]
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(word)
            try:
                conf = int(data['conf'][i])
                if conf >= 0:
                    word_confidences.append((word, conf))
            except:
                continue

  
    full_text = "\n".join([" ".join(lines[k]) for k in sorted(lines.keys())])
    confidences = [conf for _, conf in word_confidences]
    return full_text, confidences, word_confidences

def extract_text_from_pdf(pdf_path: str):
    """
    Convert PDF pages to images and extract text with confidence from each page.
    """
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    all_confidences = []
    all_word_confidences = []

    for i, page in enumerate(pages):
        temp_path = f"temp_page_{i}.png"
        page.save(temp_path, "PNG")
        text, confs, word_confs = extract_text_from_image(temp_path)
        full_text += text + "\n"
        all_confidences.extend(confs)
        all_word_confidences.extend(word_confs)
        
        for _ in range(5):
            try:
                os.remove(temp_path)
                break
            except PermissionError:
                time.sleep(0.2)

    return full_text.strip(), all_confidences, all_word_confidences