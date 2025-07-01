# ======================================
# VectorIndexBuilder.py – Emotion-Weighted Task Planner Extension
# ======================================
#
# Adds an emotion-weighted task proposal based on keyword intensity in the uploaded documents.
# Friday will generate suggested tasks if emotionally significant keywords are detected.

import os
import json
import pickle
import hashlib
import argparse
import logging
from pathlib import Path
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import fitz  # PyMuPDF
import docx
import re
from collections import Counter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from core.MemoryCore import MemoryCore

UPLOAD_DIR = Path('core/knowledge_data/uploads')
PROFILE_FILE = Path('core/knowledge_data/profile_manager.json')
HASH_CACHE_FILE = Path('core/knowledge_data/memory/vector_hashes.json')
TASK_QUEUE_FILE = Path('core/knowledge_data/plans/task_queue.json')
LOG_FILE = Path('core/knowledge_data/memory/embedding.log')

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

EMOTION_KEYWORDS = {
    "pregnancy": ["pain", "risk", "anxiety", "emergency", "bleeding", "depression"],
    "general": ["danger", "urgent", "stress", "severe", "unsafe", "fear"]
}


def load_vector_path_from_profile():
    if not PROFILE_FILE.exists():
        logging.error("profile_manager.json not found.")
        exit(1)
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    current = profile.get("current_domain")
    vector_file = profile.get("profiles", {}).get(current, {}).get("vector_file")
    if not vector_file:
        logging.error(f"No vector file set for domain '{current}'")
        exit(1)
    return Path(vector_file)


def extract_text(file: Path) -> str:
    try:
        if file.suffix in ['.txt', '.md']:
            return file.read_text(encoding='utf-8')
        elif file.suffix == '.pdf':
            doc = fitz.open(file)
            return "\n".join([page.get_text() for page in doc])
        elif file.suffix == '.docx':
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Failed to read {file}: {e}")
    return ""


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_existing_hashes() -> set:
    if not HASH_CACHE_FILE.exists():
        return set()
    try:
        with open(HASH_CACHE_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except:
        return set()


def save_hashes(hash_set: set):
    HASH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(hash_set), f)


def scan_for_emotion_triggers(text: str) -> List[str]:
    all_keywords = EMOTION_KEYWORDS["pregnancy"] + EMOTION_KEYWORDS["general"]
    found = [word.lower() for word in re.findall(r'\b\w+\b', text.lower()) if word in all_keywords]
    return list(set(found))


def append_tasks_from_keywords(keywords: List[str], doc_name: str):
    if not keywords:
        return

    TASK_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TASK_QUEUE_FILE.exists():
        with open(TASK_QUEUE_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    else:
        tasks = []

    for keyword in keywords:
        task = {
            "task": f"Investigate emotional trigger '{keyword}' in document {doc_name}",
            "priority": "high",
            "emotion": keyword,
            "type": "reflection",
            "source": doc_name
        }
        tasks.append(task)
        logging.info(f"TASK_SUGGESTED: {task['task']}")

    with open(TASK_QUEUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)


def load_documents(upload_dir: Path, known_hashes: set) -> List[Dict]:
    documents = []
    for file in upload_dir.glob('*'):
        if file.suffix.lower() not in ['.txt', '.md', '.docx', '.pdf']:
            continue
        content = extract_text(file)
        if not content.strip():
            continue
        h = hash_text(content)
        if h in known_hashes:
            logging.info(f"SKIPPED: {file.name} (already indexed)")
            continue
        documents.append({
            'id': str(uuid4()),
            'filename': file.name,
            'content': content,
            'hash': h
        })
        logging.info(f"QUEUED: {file.name} (new)")
    return documents


def build_index(documents: List[Dict], index_path: Path, existing_hashes: set):
    texts = [doc['content'] for doc in documents]
    print(f"[INFO] Embedding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True)

    index_data = []
    if index_path.exists():
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

    memory = MemoryCore()

    for doc, vector in zip(documents, embeddings):
        index_data.append({
            'id': doc['id'],
            'filename': doc['filename'],
            'embedding': vector.tolist(),
            'content': doc['content']
        })

        existing_hashes.add(doc['hash'])
        logging.info(f"EMBEDDED: {doc['filename']}")

        memory.save_fact(
            key=doc['filename'],
            value=doc['content'],
            source="VectorIndexBuilder",
            metadata={"type": "knowledge_upload"}
        )
        logging.info(f"MEMORY_INJECTED: {doc['filename']}")

        emotion_triggers = scan_for_emotion_triggers(doc['content'])
        append_tasks_from_keywords(emotion_triggers, doc['filename'])

    with open(index_path, 'wb') as f:
        pickle.dump(index_data, f)
    save_hashes(existing_hashes)

    print(f"[SUCCESS] Vector index updated with {len(documents)} new document(s). Injected into memory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build vector index from uploaded documents")
    parser.add_argument('--input', type=str, default=str(UPLOAD_DIR), help='Directory containing text documents')
    args = parser.parse_args()

    upload_path = Path(args.input)
    if not upload_path.exists():
        print(f"[ERROR] Input path {upload_path} does not exist.")
        exit(1)

    known_hashes = load_existing_hashes()
    docs = load_documents(upload_path, known_hashes)
    if not docs:
        print("[INFO] No new documents to embed.")
        exit(0)

    index_output_path = load_vector_path_from_profile()
    build_index(docs, index_output_path, known_hashes)
