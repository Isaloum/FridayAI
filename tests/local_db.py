import sqlite3
conn = sqlite3.connect('friday_local.db')
print("✓ Connected to Local Database")
conn.close()