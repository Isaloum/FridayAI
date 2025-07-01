# ======================================
# auto_index_update.py – Live Brain Expansion
# ======================================
#
# This script ingests all JSON scraped research files from `scraped/`
# and appends them to FridayAI's vector index.
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
INDEX_FILE = Path("core/knowledge_data/vector_index.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


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


def embed_and_append(new_docs: List[dict]):
    """Embed each new article and append to vector index."""
    if not new_docs:
        print("[INFO] No new documents to process.")
        return

    contents = [doc['content'] for doc in new_docs]
    embeddings = model.encode(contents, show_progress_bar=True)

    # Load existing index if exists
    if INDEX_FILE.exists():
        with open(INDEX_FILE, 'rb') as f:
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
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index, f)

    print(f"[SUCCESS] Indexed and added {len(new_docs)} new articles.")


if __name__ == '__main__':
    scraped_docs = load_scraped_json()
    embed_and_append(scraped_docs)
