# ======================================
# auto_index_update.py – Live Brain Expansion (Profile-Aware)
# ======================================
#
# This script ingests all JSON scraped research files from `scraped/`
# and appends them to FridayAI's vector index defined in the active profile.
#
# CMD Usage:
# python auto_index_update.py
#
# Dependencies: sentence-transformers, uuid, pickle, os, json, pathlib

import os
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List

# Constants
SCRAPE_DIR = Path("core/knowledge_data/scraped")
PROFILE_FILE = Path("core/knowledge_data/profile_manager.json")
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def load_profile_vector_path():
    """Load vector path from active domain profile."""
    if not PROFILE_FILE.exists():
        print("[ERROR] profile_manager.json not found.")
        exit(1)
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    current = data.get("current_domain")
    vector_file = data.get("profiles", {}).get(current, {}).get("vector_file")
    if not vector_file:
        print(f"[ERROR] No vector file found in profile for domain '{current}'")
        exit(1)
    return Path(vector_file)


def load_scraped_json() -> List[dict]:
    """Load all .json articles from the scrape directory."""
    articles = []
    for file in SCRAPE_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if 'content' in data:
                    articles.append(data)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")
    return articles


def embed_and_append(new_docs: List[dict], index_file: Path):
    """Embed each new article and append to vector index."""
    if not new_docs:
        print("[INFO] No new documents to process.")
        return

    contents = [doc['content'] for doc in new_docs]
    embeddings = model.encode(contents, show_progress_bar=True)

    # Load existing index if exists
    if index_file.exists():
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
    else:
        index = []

    for doc, vec in zip(new_docs, embeddings):
        index.append({
            'id': doc.get('id'),
            'filename': doc.get('url'),
            'embedding': vec.tolist(),
            'content': doc.get('content'),
            'source': doc.get('source'),
            'timestamp': doc.get('timestamp'),
            'query': doc.get('query')
        })

    # Save updated index
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)

    print(f"[SUCCESS] Indexed and added {len(new_docs)} new articles to {index_file}.")


if __name__ == '__main__':
    scraped_docs = load_scraped_json()
    index_path = load_profile_vector_path()
    embed_and_append(scraped_docs, index_path)
