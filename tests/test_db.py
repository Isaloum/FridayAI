# =============================================
# File: C:\Users\ihabs\FridayAI\tests\test_db.py  
# Purpose: JSON-backed memory manager for saving and querying cognitive data
# Location: Must be in tests\test_db.py
# =============================================

import psycopg2

conn = psycopg2.connect(
    host="database-1.cz88ou40ccpw.us-east-2.rds.amazonaws.com",
    database="postgres",
    user="postgres",
    password="~~Pierre32Lea~~",
    port=5432
)

print("✓ Connected to Friday AI Database")
conn.close()