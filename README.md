# Marksheet Extractor – Multimodal LLM vs OCR + LLM

## 📌 Project Overview
This project is focused on **automated marksheet data extraction** using two different approaches:
1. **Multimodal LLM (OpenAI based)**
2. **OCR + LLM (Hybrid pipeline)**

The goal is to extract structured information (like student name, roll number, marks, grades, etc.) from marksheets efficiently.

---

## ⚡ Approach 1: Multimodal LLM (OpenAI)
In this approach, we directly use a **Multimodal Large Language Model (LLM)** such as OpenAI’s GPT, which can handle both text and images.

- The marksheet image is directly fed into the LLM.
- The LLM processes the image and extracts required structured fields.
- No extra OCR step is required.

✅ **Advantages**:
- **Fast & efficient** – Results are generated quickly since no intermediate step is involved.  
- **Better accuracy** – Multimodal LLMs understand both images and text naturally.  
- **Clean pipeline** – Direct extraction without additional libraries.  

❌ **Disadvantages**:
- **Requires API access** – Dependent on OpenAI or similar multimodal APIs.  
- **Paid usage** – API usage might be costly at scale.  
- **Internet dependency** – Needs a stable internet connection.  

---

## ⚡ Approach 2: OCR + LLM
In this hybrid method:
1. First, the marksheet image is passed through an **OCR (Optical Character Recognition)** engine such as **Tesseract** or **EasyOCR**.  
2. The extracted raw text is then given to the **LLM** for parsing and structuring.  

✅ **Advantages**:
- **Can run locally** – No need for cloud multimodal APIs if OCR + local LLM are used.  
- **More flexible** – OCR can be fine-tuned or replaced with better models if needed.  

❌ **Disadvantages**:
- **Slower performance** – Since OCR runs first and then LLM processes the output, the pipeline is relatively slow.  
- **Hardware dependency** – On machines without GPU, OCR takes significantly more time.  
- **Error-prone** – OCR sometimes introduces mistakes in text extraction, which reduces the LLM’s accuracy.  
- **Lengthy process** – Multiple steps increase complexity.  

---

## 🚀 Why Multimodal LLM is Preferred?
- **Speed**: OpenAI’s multimodal LLM gives results much faster than OCR + LLM.  
- **Accuracy**: It avoids OCR-induced errors.  
- **Simplicity**: A single model handles both vision and text, reducing complexity.  

However, **OCR + LLM** can still be useful in environments where:
- API access is restricted,  
- Internet is unavailable, or  
- Cost needs to be minimized.  

---

## 📂 Project Structure
