# test_intent_detector.py
from IntentDetector import IntentDetector

detector = IntentDetector()

test_inputs = [
    "Can you remind me to journal tomorrow?",
    "I can't take this pain anymore.",
    "Tell me what you remember about me.",
    "Today I felt anxious and disconnected.",
    "How have I been feeling lately?",
    "What is reinforcement learning?",
    "My shoulder is worse again today."
]

for input_text in test_inputs:
    result = detector.detect_intent(input_text)
    print(f"\n🗣️ Input: {input_text}")
    print(f"🧠 Detected Intent: {result['intent']} (confidence: {result['confidence']})")
    if result['matched_pattern']:
        print(f"🔎 Matched Pattern: {result['matched_pattern']}")
