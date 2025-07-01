# =========================================
# File: upload_config.py
# Purpose: Configuration for file ingestion rules and paths
# =========================================

import os
from datetime import datetime

# 🔧 Base folder where uploaded knowledge files will be saved
BASE_DIR = os.path.abspath(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd())
UPLOAD_FOLDER = os.path.join(BASE_DIR, "knowledge_data", "uploads")

# ✅ Supported file types and how to parse them
UPLOAD_CONFIG = {
    "knowledge_directory": UPLOAD_FOLDER,  # Safe, absolute path now

    # Extensions that are allowed for ingestion
    "supported_extensions": [".txt", ".md", ".pdf", ".docx"],

    # Type of parser to apply for each supported extension
    "parsers": {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
        ".docx": "docx"
    },

    # Enable or disable deduplication before ingesting
    "deduplicate": True
}

# 📌 Metadata generator for uploaded files (expandable in future)
def generate_metadata(filename: str, domain: str = "general") -> dict:
    return {
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat(),
        "domain": domain,
        "verified": False  # Can be toggled manually after review
    }

# 🧪 File validator to check for supported file types
def is_allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in UPLOAD_CONFIG["supported_extensions"]
