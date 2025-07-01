# =============================================
# File: PregnancyKnowledgeSummarizer.py
# Purpose: Extract, summarize, and tag PDF knowledge for ingestion
# =============================================

import os
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from core.MemoryCore import MemoryCore
from transformers import pipeline

# === Configure Paths ===
DOCS_PATH = Path("docs")
SUMMARY_TAG = "pregnancy_knowledge"
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class PregnancyKnowledgeSummarizer:
    def __init__(self):
        self.memory = MemoryCore()

    def extract_text_from_pdf(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            print(f"[ERROR] Failed to read {pdf_path.name}: {e}")
            return ""

    def summarize_text(self, text, max_chunk_len=2000):
        chunks = [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
        return "\n".join(summaries)

    def process_all_pdfs(self):
        print("[INFO] Starting PDF summarization from docs/")
        for file in DOCS_PATH.glob("*.pdf"):
            print(f"[PROCESSING] {file.name}")
            raw_text = self.extract_text_from_pdf(file)
            if not raw_text:
                continue
            summary = self.summarize_text(raw_text)
            memory_entry = {
                "type": "knowledge",
                "source": file.name,
                "tags": [SUMMARY_TAG],
                "timestamp": datetime.now().isoformat(),
                "content": summary
            }
            self.memory.save_memory(memory_entry)
            print(f"✅ Summary injected into memory: {file.name}")


if __name__ == '__main__':
    summarizer = PregnancyKnowledgeSummarizer()
    summarizer.process_all_pdfs()
