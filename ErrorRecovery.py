# ErrorRecovery.py

class ErrorRecovery:
    @staticmethod
    def validate_fact(fact, known_facts):
        if fact not in known_facts.values():
            return None  # Fact mismatch detected
        return fact

    @staticmethod
    def recover_missing_fact(requested_fact):
        return f"Fact '{requested_fact}' not found. Please update or specify."
