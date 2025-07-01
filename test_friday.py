# =====================================
# File: test_friday.py
# Purpose: Test Friday's real AI brain
# =====================================

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLMCore import LLMCore

def test_friday_ai():
    print("🧠 Testing Friday's AI Brain...")
    
    # Initialize LLMCore
    llm = LLMCore()
    
    print(f"OpenAI Available: {llm.openai_available}")
    print(f"API Enabled: {llm.enabled}")
    
    if not llm.enabled:
        print("❌ OpenAI not configured. Check your API key.")
        return
    
    # Health check
    health = llm.health_check()
    print(f"Health Status: {health['status']}")
    
    if health['status'] != 'healthy':
        print(f"❌ Health check failed: {health}")
        return
    
    print("\n✅ Friday's AI brain is working!")
    print("🤱 Testing pregnancy conversation...")
    
    # Test pregnancy conversation
    test_cases = [
        {
            "input": "I'm 20 weeks pregnant and feeling overwhelmed about becoming a mom. Is this normal?",
            "emotional_context": {
                "primary_emotion": "anxiety",
                "intensity": 7,
                "sentiment": "negative"
            },
            "user_profile": {
                "pregnancy_week": 20,
                "trimester": "second",
                "concerns": ["first_time_mom", "anxiety"]
            }
        },
        {
            "input": "I felt the baby kick for the first time today! I'm so excited!",
            "emotional_context": {
                "primary_emotion": "joy",
                "intensity": 9,
                "sentiment": "positive"
            },
            "user_profile": {
                "pregnancy_week": 18,
                "trimester": "second"
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"You: {test['input']}")
        
        response = llm.generate_response(
            test['input'],
            test['emotional_context'],
            [],  # No memory context for test
            test['user_profile']
        )
        
        if response['success']:
            print(f"Friday: {response['reply']}")
            print(f"✅ Success! Tokens used: {response['metadata'].get('tokens_used', 0)}")
        else:
            print(f"❌ Failed: {response}")
    
    print(f"\n🎉 Test complete! Conversation history: {len(llm.conversation_history)} messages")

if __name__ == "__main__":
    test_friday_ai()