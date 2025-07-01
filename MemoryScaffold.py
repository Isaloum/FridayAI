# =========================================
# File: MemoryScaffold.py
# Purpose: Persistent memory handling for Friday
# =========================================

import os
import json
import chromadb
from datetime import datetime

# === Load Friday's persistent memory collection ===
MEMORY_PATH = "memory_store"

client = chromadb.PersistentClient(path=MEMORY_PATH)
memory_collection = client.get_or_create_collection(name="friday_core_memory")

# === Search past memory entries ===
def search_memory(query: str, top_k: int = 10):
    results = memory_collection.query(query_texts=[query], n_results=top_k)
    raw = results.get("documents", [[]])[0]
    
    parsed = []
    for doc in raw:
        try:
            parsed.append(json.loads(doc))
        except json.JSONDecodeError:
            continue  # skip malformed entries
    return parsed

# === Save new memory entry ===
def store_memory(user_id: str, data: dict, metadata: dict = None):
    if not metadata:
        metadata = {"source": "MemoryScaffold"}

    memory_id = f"{user_id}-{datetime.now().isoformat()}"
    content = json.dumps(data)  # ✅ ensure valid JSON format

    memory_collection.add(
        documents=[content],
        ids=[memory_id],
        metadatas=[metadata]
    )
    return memory_id

# === Persist memory index ===
def save_memory():
    client.persist()
