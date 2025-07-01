# ======================================
# VectorIndexBuilder.py – Embedding Engine
# ======================================
#
# This module scans the upload directory for text-based documents,
# generates vector embeddings using SentenceTransformer, and stores
# them with metadata in a .pkl index file.
#
# CMD Usage:
# python VectorIndexBuilder.py --input core/knowledge_data/uploads
#
# Dependencies: sentence-transformers, uuid, pickle

import os
import json
import pickle
import argparse
from pathlib import Path
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Load sentence-transformers model (lightweight and fast)
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# Define constants for directories and output file
UPLOAD_DIR = Path('core/knowledge_data/uploads')
INDEX_FILE = Path('core/knowledge_data/vector_index.pkl')


def load_documents(upload_dir: Path) -> List[Dict]:
    """
    Loads all text-based documents from the specified directory.
    Only includes .txt, .md, .docx, .pdf, .json extensions.
    Returns a list of document dicts with ID, filename, and content.
    """
    documents = []
    for file in upload_dir.glob('*'):
        if file.suffix not in ['.txt', '.md', '.docx', '.pdf', '.json']:
            continue
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            documents.append({
                'id': str(uuid4()),
                'filename': file.name,
                'content': content
            })
        except Exception as e:
            print(f"[ERROR] Failed to read {file}: {e}")
    return documents


def build_index(documents: List[Dict]):
    """
    Takes a list of documents and generates vector embeddings.
    Stores each document's embedding, filename, content, and ID in a pickle file.
    """
    texts = [doc['content'] for doc in documents]
    print(f"[INFO] Embedding {len(texts)} documents...")

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # Construct data with embeddings
    index_data = []
    for doc, vector in zip(documents, embeddings):
        index_data.append({
            'id': doc['id'],
            'filename': doc['filename'],
            'embedding': vector.tolist(),
            'content': doc['content']
        })

    # Save to pickle file
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"[SUCCESS] Vector index saved to {INDEX_FILE}")


if __name__ == '__main__':
    # Command-line interface setup
    parser = argparse.ArgumentParser(description="Build vector index from uploaded documents")
    parser.add_argument('--input', type=str, default=str(UPLOAD_DIR), help='Directory containing text documents')
    args = parser.parse_args()

    # Validate path
    upload_path = Path(args.input)
    if not upload_path.exists():
        print(f"[ERROR] Input path {upload_path} does not exist.")
        exit(1)

    # Load and process documents
    docs = load_documents(upload_path)
    if not docs:
        print("[WARN] No documents found to index.")
        exit(0)

    build_index(docs)
