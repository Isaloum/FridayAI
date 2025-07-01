# test_personality.py
from PersonalityCore import PersonalityCore

# Sample emotional trend input (simulate the last 7 days)
trend_data = [
    {"date": "2025-05-06", "dominant": "sad"},
    {"date": "2025-05-07", "dominant": "anxious"},
    {"date": "2025-05-08", "dominant": "angry"},
    {"date": "2025-05-09", "dominant": "anxious"},
    {"date": "2025-05-10", "dominant": "happy"},
]

# Initialize personality engine
profile = PersonalityCore()

# Update traits based on emotional trend
updated = profile.update_traits(trend_data)

# Print results
print("🎭 Updated Trait Profile:")
for trait, value in updated.items():
    print(f"{trait}: {value}")

# Generate tone descriptor
tone = profile.get_tone_description()
print(f"\n🗣️ Tone Summary: {tone}")
