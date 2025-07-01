from EmotionCore import EmotionCore

emotion_core = EmotionCore()

# Sample inputs to test all emotion categories
test_inputs = [
    "I'm really happy and grateful for today.",
    "I feel so sad and hopeless.",
    "I'm angry and frustrated with everything.",
    "This is terrifying — I'm so scared.",
    "I love spending time with you.",
    "I feel ashamed and guilty about what I said.",
    "I don't know what's wrong. I feel numb.",
    "I need someone to talk to.",
    "I'm confused and overwhelmed.",
    "You make me feel seen and valued."
]

for text in test_inputs:
    emotions = emotion_core.analyze(text)
    dominant = emotion_core.get_dominant_emotion(emotions)
    print(f"\n🗣️ Input: {text}")
    print(f"🔍 Detected: {emotions}")
    print(f"🏷️ Dominant Emotion: {dominant}")
    print("-" * 60)
