# =====================================
# TransportCore.py - Original Transport Logic
# =====================================
from typing import Dict

class TransportCore:
    """Original transportation knowledge handler"""
    
    def __init__(self):
        self.knowledge_base = {
            'yul_transport': {
                'response': (
                    "From Laval to YUL Airport:\n"
                    "1. Taxi/Uber: 40-60$ CAD (35-50 mins)\n"
                    "2. 747 Bus: Lionel-Groulx metro (2.75$)\n"
                    "3. Airport Shuttle: 1-800-123-4567 (65$+)\n"
                    "4. Car Rental: Available at YUL"
                ),
                'keywords': ['yul', 'airport', 'transport', 'bus']
            }
        }

    def handle_query(self, query: str) -> Dict:
        """Original transport query handling"""
        query = query.lower()
        best_match = {'content': None, 'confidence': 0.0}
        
        for entry in self.knowledge_base.values():
            matches = sum(1 for kw in entry['keywords'] if kw in query)
            confidence = matches / len(entry['keywords'])
            
            if confidence > best_match['confidence']:
                best_match.update({
                    'content': entry['response'],
                    'confidence': confidence
                })
        
        return {'domain': 'transport', **best_match}