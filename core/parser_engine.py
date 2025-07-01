# =====================================
# File: parser_engine.py
# Purpose: Extracts text from supported knowledge files (PDF, TXT, MD)
# =====================================

import os
import fitz  # PyMuPDF
import logging
from upload_config import ALLOWED_EXTENSIONS, KNOWLEDGE_DIR, DEDUPLICATE

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logging.error(f"Failed to read PDF: {file_path} | Error: {e}")
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read TXT: {file_path} | Error: {e}")
        return ""

def extract_text_from_md(file_path):
    return extract_text_from_txt(file_path)  # Same as .txt

def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return list(set(lines)) if DEDUPLICATE else lines

def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        raw = extract_text_from_pdf(file_path)
    elif ext == '.txt':
        raw = extract_text_from_txt(file_path)
    elif ext == '.md':
        raw = extract_text_from_md(file_path)
    else:
        logging.warning(f"Unsupported file skipped: {file_path}")
        return []

    return clean_text(raw)

def parse_all_files():
    parsed_data = {}
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                full_path = os.path.join(root, fname)
                logging.info(f"Parsing: {full_path}")
                parsed_data[fname] = parse_file(full_path)
    return parsed_data

if __name__ == '__main__':
    data = parse_all_files()
    print(f"Loaded files: {list(data.keys())}")
