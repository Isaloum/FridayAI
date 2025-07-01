from datasets import load_dataset
import os

# Load Dolly 15K dataset
dataset = load_dataset("databricks/databricks-dolly-15k")["train"]

# Create /data folder if not exists
os.makedirs("data", exist_ok=True)

# Write into a plain .txt file
with open("data/dolly_qa.txt", "w", encoding="utf-8") as out:
    for item in dataset:
        question = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        answer = item.get("response", "").strip()
        if not (question and answer):
            continue

        out.write(f"Q: {question}\n")
        if context:
            out.write(f"Context: {context}\n")
        out.write(f"A: {answer}\n\n")

print("✅ dolly_qa.txt created in /data/")
