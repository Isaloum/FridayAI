"""
db_repair.py
------------
Checks and patches the ChromaDB SQLite schema for missing 'topic' column.
"""

import sqlite3
import os

DB_PATH = "memory/chroma.sqlite3"

def patch_db_schema():
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] DB file not found: {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(collections);")
        columns = [col[1] for col in cursor.fetchall()]

        if "topic" not in columns:
            print("[INFO] 'topic' column missing. Patching...")
            cursor.execute("ALTER TABLE collections ADD COLUMN topic TEXT;")
            conn.commit()
            print("[SUCCESS] Column 'topic' added to 'collections' table.")
        else:
            print("[OK] 'topic' column already exists. No action needed.")
    except Exception as e:
        print(f"[ERROR] Failed to patch DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    patch_db_schema()
