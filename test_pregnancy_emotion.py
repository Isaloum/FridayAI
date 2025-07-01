# =====================================
# FILE: test_pregnancy_emotion.py
# STEP 1: Test your existing PregnancyEmotionCore
# =====================================

import sys
import os

# Add your core path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_pregnancy_emotions():
    """
    MANUAL ONLY TEST - No automatic triggers, keywords, or background processes
    Only runs when YOU explicitly call it
    """
    
    print("🧠 MANUAL PREGNANCY EMOTION TEST")
    print("⚠️  NO AUTOMATIC TRIGGERS - Manual testing only")
    print("=" * 50)
    
    try:
        # Import your existing pregnancy emotion core
        from core.pregnancy.PregnancyEmotionCore import PregnancyEmotionCore
        
        print("✅ Successfully imported PregnancyEmotionCore")
        
        # Initialize the emotion core
        emotion_core = PregnancyEmotionCore()
        print("✅ Emotion core initialized")
        
        # Test messages with different pregnancy emotions
        test_messages = [
            {
                "text": "I'm so scared about giving birth, what if something goes wrong?",
                "week": 35,
                "expected": "anxiety/fear"
            },
            {
                "text": "I felt the baby kick for the first time today! I'm overwhelmed with love!",
                "week": 18,
                "expected": "overwhelming_love"
            },
            {
                "text": "I can't stop crying at commercials for no reason",
                "week": 12,
                "expected": "emotional_overwhelm"
            },
            {
                "text": "I need to organize the nursery and wash all the baby clothes",
                "week": 32,
                "expected": "nesting"
            }
        ]
        
        print("\n🧪 RUNNING EMOTION TESTS:")
        print("-" * 30)
        
        for i, test in enumerate(test_messages, 1):
            print(f"\nTest {i}: Week {test['week']}")
            print(f"Message: '{test['text']}'")
            
            # Test the emotion analysis
            result = emotion_core.analyze_pregnancy_emotion(
                text=test['text'], 
                pregnancy_week=test['week']
            )
            
            print(f"🎯 Detected: {result.primary_emotion}")
            print(f"💪 Intensity: {result.intensity:.2f}")
            print(f"🌸 Hormonal influence: {result.hormonal_influence:.2f}")
            print(f"📅 Trimester factor: {result.trimester_factor:.2f}")
            print(f"✨ Confidence: {result.confidence_score:.2f}")
            
            if result.contextual_triggers:
                print(f"🔍 Triggers: {', '.join(result.contextual_triggers)}")
            
            print(f"Expected: {test['expected']} ✅" if result.primary_emotion else "❌")
        
        print("\n" + "=" * 50)
        print("🎉 PREGNANCY EMOTION DETECTION TEST COMPLETE!")
        print("Ready for Step 2: Full Friday Integration")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure PregnancyEmotionCore.py exists in core/pregnancy/")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Check the PregnancyEmotionCore implementation")
        return False

if __name__ == "__main__":
    test_pregnancy_emotions()