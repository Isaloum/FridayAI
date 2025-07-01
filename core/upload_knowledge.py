# =====================================
# File: upload_knowledge.py
# Purpose: Upload and store knowledge documents into FridayAI brain
# =====================================

import argparse
import os
import shutil
#from core.upload_config import UPLOAD_FOLDER, generate_metadata, is_allowed_file
from upload_config import UPLOAD_FOLDER, generate_metadata, is_allowed_file

def main():
    parser = argparse.ArgumentParser(description="Upload knowledge documents into FridayAI system.")
    parser.add_argument("--file", type=str, required=True, help="Path to the document to upload.")
    parser.add_argument("--domain", type=str, default="general", help="Domain category for the document.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    filepath = args.file
    filename = os.path.basename(filepath)

    print(f"[INFO] Checking file: {filepath}")
    if not os.path.exists(filepath):
        print("[ERROR] File does not exist.")
        return

    if not is_allowed_file(filename):
        print("[ERROR] File type not allowed.")
        return

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    destination = os.path.join(UPLOAD_FOLDER, filename)

    try:
        shutil.copyfile(filepath, destination)
        metadata = generate_metadata(filename=filename, domain=args.domain)

        if args.verbose:
            print(f"[UPLOAD] File copied to: {destination}")
            print(f"[UPLOAD] Metadata: {metadata}")
        print("[SUCCESS] Upload completed.")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    main()
