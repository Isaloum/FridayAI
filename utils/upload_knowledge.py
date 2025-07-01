# =====================================
# File: upload_knowledge.py
# Purpose: Upload and embed documents into FridayAI's vector knowledge base
# =====================================

import os
import pickle
import argparse
import textract
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_SIZE = 500  # Number of characters per chunk
VECTOR_STORE_PATH = "vector_stores"
METADATA_STORE_PATH = "metadata_stores"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(METADATA_STORE_PATH, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def load_text_from_file(filepath):
    try:
        text = textract.process(filepath).decode("utf-8")
        return text
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return ""

def embed_and_store(directory, domain):
    all_chunks = []
    metadata = []

    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Processing documents"):
            if not file.lower().endswith((".pdf", ".txt", ".docx")):
                continue
            full_path = os.path.join(root, file)
            text = load_text_from_file(full_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            metadata.extend([{"source": file, "domain": domain, "chunk_id": i} for i in range(len(chunks))])

    if not all_chunks:
        print("[WARN] No valid content found to embed.")
        return

    embeddings = model.encode(all_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, f"{VECTOR_STORE_PATH}/{domain}.index")
    with open(f"{METADATA_STORE_PATH}/{domain}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"[DONE] Uploaded and embedded {len(all_chunks)} chunks under domain: {domain}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload knowledge documents into FridayAI's brain.")
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--domain", required=True, help="Domain name to tag the documents (e.g., pregnancy, law)")
    args = parser.parse_args()

    embed_and_store(args.dir, args.domain)
