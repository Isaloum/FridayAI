# PersonalDataCore.py
class PersonalDataCore:
    """Dynamically handles personal data using MemoryCore"""
    
    PERSONAL_DATA_MAP = {
        'location': ['where am i', 'my location', 'current city'],
        'age': ['how old am i', 'my age', 'what is my age'],
        'name': ['what is my name', 'who am i']
        # Add more mappings as needed
    }

    def __init__(self, memory_core):
        self.memory = memory_core

    def is_personal_query(self, user_input):
        """Check if query requires personal data"""
        lower_input = user_input.lower()
        return any(
            trigger in lower_input
            for triggers in self.PERSONAL_DATA_MAP.values()
            for trigger in triggers
        )

    def handle_personal_query(self, user_input):
        """Process personal data requests"""
        lower_input = user_input.lower()
        
        # Find matching data type
        for data_type, triggers in self.PERSONAL_DATA_MAP.items():
            if any(trigger in lower_input for trigger in triggers):
                stored_data = self.memory.get_fact(f"user_{data_type}")
                
                if stored_data:
                    return f"Your {data_type} is {stored_data}"
                else:
                    return f"Please tell me your {data_type} first."
        
        return None