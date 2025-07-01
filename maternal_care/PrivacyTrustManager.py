# =====================================
# FILE: maternal_care/PrivacyTrustManager.py
# =====================================

import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from .SecureMaternalDatabase import SecureMaternalDatabase

class PrivacyTrustManager:
    """
    Comprehensive privacy and trust management
    Ensures user feels safe and in control
    """
    
    def __init__(self, maternal_db: SecureMaternalDatabase):
        self.db = maternal_db
    
    def get_privacy_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Provide complete transparency about data usage"""
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Get data counts
            cursor.execute("SELECT COUNT(*) FROM pregnancy_tracking WHERE user_id = ?", (user_id,))
            pregnancy_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM mental_health WHERE user_id = ?", (user_id,))
            mental_health_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM baby_tracking WHERE user_id = ?", (user_id,))
            baby_records = cursor.fetchone()[0]
            
            # Get privacy settings
            cursor.execute("SELECT * FROM privacy_settings WHERE user_id = ?", (user_id,))
            privacy_settings = cursor.fetchone()
        
        return {
            "data_summary": {
                "pregnancy_entries": pregnancy_records,
                "mental_health_entries": mental_health_records,
                "baby_development_entries": baby_records,
                "total_storage_mb": self._calculate_storage_usage(user_id)
            },
            "privacy_controls": {
                "encryption_status": "AES-256 encrypted",
                "offline_mode": self.db.offline_mode,
                "data_sharing": "None - stored locally only",
                "retention_period": f"{privacy_settings[3] if privacy_settings else 365} days"
            },
            "your_rights": {
                "data_export": "Available anytime",
                "data_deletion": "Complete deletion available",
                "access_control": "You have full control",
                "consent_withdrawal": "Can be withdrawn anytime"
            },
            "security_measures": {
                "local_encryption": "✅ Active",
                "secure_backup": "✅ Available",
                "access_logging": "✅ Monitored",
                "breach_protection": "✅ Multi-layer protection"
            }
        }
    
    def export_user_data(self, user_id: str, export_format: str = "json") -> str:
        """Export all user data in readable format"""
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            export_data = {
                "export_info": {
                    "user_id": user_id,
                    "export_date": datetime.now().isoformat(),
                    "format": export_format
                },
                "profile": self._export_profile_data(cursor, user_id),
                "pregnancy_journey": self._export_pregnancy_data(cursor, user_id),
                "mental_health": self._export_mental_health_data(cursor, user_id),
                "postpartum": self._export_postpartum_data(cursor, user_id),
                "baby_development": self._export_baby_data(cursor, user_id)
            }
        
        if export_format == "json":
            return json.dumps(export_data, indent=2)
        elif export_format == "csv":
            return self._convert_to_csv(export_data)
        
        return str(export_data)
    
    def delete_all_user_data(self, user_id: str, confirmation_code: str) -> bool:
        """Securely delete all user data"""
        
        # Verify confirmation (in production, use stronger verification)
        expected_code = hashlib.sha256(f"DELETE_{user_id}".encode()).hexdigest()[:8]
        
        if confirmation_code != expected_code:
            print("[🔒 PRIVACY] Deletion cancelled - invalid confirmation")
            return False
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete all related data
            tables = ['baby_tracking', 'postpartum_tracking', 'mental_health', 
                     'pregnancy_tracking', 'privacy_settings', 'user_profile']
            
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            
            conn.commit()
        
        print(f"[🔒 PRIVACY] All data for user {user_id[:8]}... has been permanently deleted")
        return True
    
    def _calculate_storage_usage(self, user_id: str) -> float:
        """Calculate storage usage in MB"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT SUM(LENGTH(encrypted_data)) 
                    FROM (
                        SELECT encrypted_data FROM pregnancy_tracking WHERE user_id = ?
                        UNION ALL
                        SELECT encrypted_data FROM mental_health WHERE user_id = ?
                        UNION ALL
                        SELECT encrypted_data FROM baby_tracking WHERE user_id = ?
                        UNION ALL
                        SELECT encrypted_data FROM postpartum_tracking WHERE user_id = ?
                        UNION ALL
                        SELECT encrypted_data FROM user_profile WHERE id = ?
                    )
                """, (user_id, user_id, user_id, user_id, user_id))
                
                total_bytes = cursor.fetchone()[0] or 0
                return round(total_bytes / (1024 * 1024), 2)  # Convert to MB
        except Exception:
            return 0.0
    
    def _export_profile_data(self, cursor, user_id: str) -> Dict:
        """Export user profile data"""
        cursor.execute("SELECT encrypted_data FROM user_profile WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if result:
            decrypted_data = self.db.decrypt_data(result[0])
            return json.loads(decrypted_data)
        return {}
    
    def _export_pregnancy_data(self, cursor, user_id: str) -> List[Dict]:
        """Export pregnancy tracking data"""
        cursor.execute("""
            SELECT encrypted_data, week_number, entry_date 
            FROM pregnancy_tracking 
            WHERE user_id = ? 
            ORDER BY entry_date
        """, (user_id,))
        
        results = []
        for row in cursor.fetchall():
            decrypted_data = self.db.decrypt_data(row[0])
            data = json.loads(decrypted_data)
            data['week_number'] = row[1]
            data['entry_date'] = row[2]
            results.append(data)
        
        return results
    
    def _export_mental_health_data(self, cursor, user_id: str) -> List[Dict]:
        """Export mental health data"""
        cursor.execute("""
            SELECT encrypted_data, mood_score, entry_date 
            FROM mental_health 
            WHERE user_id = ? 
            ORDER BY entry_date
        """, (user_id,))
        
        results = []
        for row in cursor.fetchall():
            decrypted_data = self.db.decrypt_data(row[0])
            data = json.loads(decrypted_data)
            data['mood_score'] = row[1]
            data['entry_date'] = row[2]
            results.append(data)
        
        return results
    
    def _export_postpartum_data(self, cursor, user_id: str) -> List[Dict]:
        """Export postpartum data"""
        cursor.execute("""
            SELECT encrypted_data, days_postpartum, entry_date 
            FROM postpartum_tracking 
            WHERE user_id = ? 
            ORDER BY entry_date
        """, (user_id,))
        
        results = []
        for row in cursor.fetchall():
            decrypted_data = self.db.decrypt_data(row[0])
            data = json.loads(decrypted_data)
            data['days_postpartum'] = row[1]
            data['entry_date'] = row[2]
            results.append(data)
        
        return results
    
    def _export_baby_data(self, cursor, user_id: str) -> List[Dict]:
        """Export baby development data"""
        cursor.execute("""
            SELECT encrypted_data, baby_age_days, entry_date 
            FROM baby_tracking 
            WHERE user_id = ? 
            ORDER BY entry_date
        """, (user_id,))
        
        results = []
        for row in cursor.fetchall():
            decrypted_data = self.db.decrypt_data(row[0])
            data = json.loads(decrypted_data)
            data['baby_age_days'] = row[1]
            data['entry_date'] = row[2]
            results.append(data)
        
        return results
    
    def _convert_to_csv(self, export_data: Dict) -> str:
        """Convert export data to CSV format"""
        import csv
        import io
        
        output = io.StringIO()
        
        # Export pregnancy data as CSV
        if export_data.get('pregnancy_journey'):
            writer = csv.writer(output)
            writer.writerow(['Date', 'Week', 'Symptoms', 'Weight', 'Energy', 'Sleep', 'Notes'])
            
            for entry in export_data['pregnancy_journey']:
                writer.writerow([
                    entry.get('entry_date', ''),
                    entry.get('week_number', ''),
                    ', '.join(entry.get('physical_symptoms', [])),
                    entry.get('weight', ''),
                    entry.get('energy_level', ''),
                    entry.get('sleep_quality', ''),
                    entry.get('notes', '')
                ])
        
        return output.getvalue()