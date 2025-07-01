from rapidfuzz import fuzz

class KnowledgeCore:
    def __init__(self):
        self.knowledge_db = {
            "yul_airport_transport_laval": {
                "response": (
                    "From Laval to YUL Airport:\n"
                    "1. Taxi/Uber: ~$40–$60 (35–50 mins)\n"
                    "2. 747 Express Bus: Metro to Lionel-Groulx, then 747 bus ($2.75)\n"
                    "3. Communauto car-sharing: Hourly rentals\n"
                    "4. Airport shuttle: Some Laval hotels offer shuttles (~$65+)"
                ),
                "keywords": {"laval", "yul", "airport", "transport", "ride", "bus", "taxi", "uber"}
            },
            "emergency_number_canada": {
                "response": "In Canada, the emergency number is 911 for police, fire, or medical assistance.",
                "keywords": {"emergency", "911", "canada", "help"}
            }
        }

    def lookup(self, query):
        clean_query = query.lower()
        for key, entry in self.knowledge_db.items():
            matched_keywords = sum(
                1 for kw in entry["keywords"] if fuzz.partial_ratio(kw, clean_query) > 80
            )
            if matched_keywords >= 3:
                return entry["response"]
        return None
