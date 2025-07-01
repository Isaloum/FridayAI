# =========================================
# File: ReflectionEngine.py
# Purpose: Analyze Friday's memory and generate daily reflections
# =========================================

import os
import json
import chromadb
from datetime import datetime, timedelta

# === Load Friday's persistent memory collection ===
MEMORY_PATH = "memory_store"

client = chromadb.PersistentClient(path="memory_store")

memory_collection = client.get_or_create_collection(name="friday_core_memory")

# === Search past memory entries ===
def search_memory(query: str, top_k: int = 10):
    results = memory_collection.query(query_texts=[query], n_results=top_k)
    raw = results.get("documents", [[]])[0]
    return [json.loads(doc) for doc in raw if doc]

# === Save new memory entry ===
def store_memory(user_id: str, data: dict, metadata: dict = None):
    if not metadata:
        metadata = {"source": "ReflectionEngine"}

    memory_id = f"{user_id}-{datetime.now().isoformat()}"
    content = json.dumps(data)
    memory_collection.add(
        documents=[content],
        ids=[memory_id],
        metadatas=[metadata]
    )
    return memory_id

# === Generate daily reflective summary ===
def generate_daily_reflection(user_id: str = "user"):
    query = "summarize last 24 hours of user emotion and goals"
    recent = search_memory(query=query, top_k=10)

    combined = "\n".join([
        f"- Input: {entry.get('input')}\n   Reply: {entry.get('reply')}\n   Emotion: {entry.get('emotion')} | Intent: {entry.get('intent')}"
        for entry in recent if isinstance(entry, dict)
    ])

    if not combined:
        return "No significant user activity found today."

    summary = f"?? Daily Reflection ({datetime.now().date()}):\n"
    summary += f"- User focused on: {query}\n"
    summary += f"- Detected themes:\n{combined}"

    store_memory(user_id, {
        "type": "daily_reflection",
        "summary": summary,
        "time": datetime.now().isoformat()
    }, metadata={"source": "ReflectionEngine"})

    return summary
