# =====================================
# File: core/KnowledgeUnit.py - CLEAN KNOWLEDGE SYSTEM
# Purpose: Reads local documents and returns relevant citations (NO POLLUTION)
# Location: core/ folder
# =====================================

import os
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "memory/knowledge.index"
META_PATH = "memory/knowledge_meta.pkl"
DOCS_PATH = "docs/"

# 🚫 FILES TO EXCLUDE FROM KNOWLEDGE (PREVENTS POLLUTION)
EXCLUDED_FILES = {
    'requirements.txt', 
    'cognition_notes.txt', 
    '.gitignore', 
    '.env',
    'config.txt',
    'setup.txt',
    'install.txt',
    'readme.txt',
    'launch.txt',
    'memory.key'
}

def _load_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    return None, []

def _save_index(index, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def _split_text(text: str, max_len=500) -> List[str]:
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind(".", 0, max_len)
        if split_at == -1: split_at = max_len
        chunks.append(text[:split_at+1].strip())
        text = text[split_at+1:].strip()
    if text:
        chunks.append(text)
    return chunks

def _is_valid_knowledge_file(filename: str) -> bool:
    """Filter out junk files from knowledge base"""
    return filename.lower() not in EXCLUDED_FILES

def ingest_documents():
    index = faiss.IndexFlatL2(384)
    metadata = []
    
    # GET ONLY CLEAN KNOWLEDGE FILES
    all_files = list(Path(DOCS_PATH).glob("*.txt"))
    valid_files = [f for f in all_files if _is_valid_knowledge_file(f.name)]
    
    # Silent operation - only show results
    for path in valid_files:
        text = Path(path).read_text(encoding="utf-8")
        chunks = _split_text(text)

        embeddings = MODEL.encode(chunks)
        index.add(embeddings)

        for chunk in chunks:
            metadata.append({
                "text": chunk,
                "source": path.name
            })

    _save_index(index, metadata)
    print(f"[KNOWLEDGE] Clean knowledge loaded: {len(metadata)} chunks from {len(valid_files)} files")

def query_knowledge(question: str, top_k=3) -> List[Dict]:
    index, meta = _load_index()
    if index is None:
        return [{"error": "No knowledge index found. Run ingest_documents() first."}]

    q_embed = MODEL.encode([question])
    D, I = index.search(q_embed, top_k)

    # FILTER OUT ANY REMAINING JUNK
    results = []
    for i in I[0]:
        if i < len(meta):
            source_file = meta[i].get('source', '')
            if _is_valid_knowledge_file(source_file):
                results.append(meta[i])
    
    return results