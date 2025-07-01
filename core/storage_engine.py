# =============================================
# File: storage_engine.py
# Purpose: Handle saving and indexing knowledge data
# Author: FridayAI Core Team
# =============================================

import os
import json
from datetime import datetime
from typing import Dict

STORAGE_DIR = "core/data_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def save_knowledge_item(title: str, content: str, source: str = "user_upload") -> str:
    """
    Saves a single knowledge item as a JSON file with a timestamp.

    Args:
        title (str): Title of the document or topic.
        content (str): Cleaned text content.
        source (str): Origin of the knowledge item (e.g., filename or user input).

    Returns:
        str: File path of the saved JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.replace(" ", "_").replace("/", "_")
    filename = f"{safe_title}_{timestamp}.json"
    file_path = os.path.join(STORAGE_DIR, filename)

    data = {
        "title": title,
        "content": content,
        "source": source,
        "timestamp": timestamp
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path

def load_all_knowledge_items() -> Dict[str, dict]:
    """
    Loads all stored knowledge items from storage.

    Returns:
        Dict[str, dict]: Dictionary of filename to document data.
    """
    knowledge_db = {}
    for filename in os.listdir(STORAGE_DIR):
        if filename.endswith(".json"):
            path = os.path.join(STORAGE_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                knowledge_db[filename] = json.load(f)
    return knowledge_db
