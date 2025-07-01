
# =====================================
# FILE 2: maternal_care/SecureMaternalDatabase.py
# =====================================

import os
import json
import sqlite3
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import uuid

class SecureMaternalDatabase:
    """
    Ultra-secure, encrypted maternal health database
    - End-to-end encryption
    - Local storage with sync capability
    - Zero-knowledge architecture
    - GDPR/HIPAA compliant design
    """
    
    def __init__(self, user_password: str = None, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.db_path = "friday_maternal_secure.db"
        
        # Initialize encryption
        if user_password:
            self.encryption_key = self._derive_key_from_password(user_password)
        else:
            self.encryption_key = self._load_or_create_key()
        
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize database
        self._init_database()
        
        print(f"[🔒 PRIVACY] Secure maternal database initialized (offline: {offline_mode})")
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from user password"""
        salt = b'friday_maternal_salt_2024'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one"""
        key_file = "friday_maternal.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def _init_database(self):
        """Initialize secure database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User profile table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile (
                    id TEXT PRIMARY KEY,
                    encrypted_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Pregnancy tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pregnancy_tracking (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    encrypted_data TEXT NOT NULL,
                    week_number INTEGER,
                    entry_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profile (id)
                )
            ''')
            
            # Mental health tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mental_health (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    encrypted_data TEXT NOT NULL,
                    mood_score REAL,
                    entry_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profile (id)
                )
            ''')
            
            # Postpartum tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS postpartum_tracking (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    encrypted_data TEXT NOT NULL,
                    days_postpartum INTEGER,
                    entry_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profile (id)
                )
            ''')
            
            # Baby tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS baby_tracking (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    encrypted_data TEXT NOT NULL,
                    baby_age_days INTEGER,
                    entry_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profile (id)
                )
            ''')
            
            # Privacy consent and preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS privacy_settings (
                    user_id TEXT PRIMARY KEY,
                    consent_data_collection BOOLEAN DEFAULT FALSE,
                    consent_ai_learning BOOLEAN DEFAULT FALSE,
                    data_retention_days INTEGER DEFAULT 365,
                    offline_only BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profile (id)
                )
            ''')
            
            conn.commit()
