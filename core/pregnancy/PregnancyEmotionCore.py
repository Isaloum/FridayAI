# =====================================
# FILE: core/pregnancy/PregnancyEmotionCore.py
# WORKING VERSION - Replace your current file with this
# =====================================

from datetime import datetime
import sys
import os

# Import your base EmotionCore
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from EmotionCoreV2 import EmotionCoreV2

class PregnancyEmotionalState:
    """Simple result container for pregnancy emotions"""
    def __init__(self, primary_emotion, intensity, hormonal_influence, 
                 trimester_factor, pregnancy_week, contextual_triggers, 
                 confidence_score):
        self.primary_emotion = primary_emotion
        self.intensity = intensity
        self.hormonal_influence = hormonal_influence
        self.trimester_factor = trimester_factor
        self.pregnancy_week = pregnancy_week
        self.contextual_triggers = contextual_triggers
        self.confidence_score = confidence_score
        self.timestamp = datetime.now().isoformat()

class PregnancyEmotionCore(EmotionCoreV2):
    """
    Enhanced emotion detection for pregnancy
    Extends your existing EmotionCoreV2
    """
    
    def __init__(self):
        super().__init__()
        print("[USING] EmotionCoreV2 from core/")
        
    def analyze_pregnancy_emotion(self, text, pregnancy_week=0, user_context=None):
        """
        Analyze pregnancy-specific emotions
        """
        
        # Use your base emotion analysis first
        try:
            base_emotions = self.analyze_emotion(text)
        except Exception as e:
            print(f"[DEBUG] Base emotion analysis failed: {e}")
            base_emotions = {"neutral": 0.5}
        
        # Simple pregnancy-specific detection
        text_lower = text.lower()
        pregnancy_emotions = {}
        
        # Detect birth anxiety
        if any(word in text_lower for word in ["scared", "afraid", "worried", "nervous"]):
            if any(word in text_lower for word in ["birth", "labor", "delivery", "giving birth"]):
                pregnancy_emotions["birth_anxiety"] = 0.8
            else:
                pregnancy_emotions["general_anxiety"] = 0.6
        
        # Detect overwhelming love
        if any(word in text_lower for word in ["love", "amazing", "incredible", "overwhelmed"]):
            if any(word in text_lower for word in ["baby", "kick", "movement", "heartbeat"]):
                pregnancy_emotions["overwhelming_love"] = 0.9
        
        # Detect emotional overwhelm
        if any(word in text_lower for word in ["crying", "emotional", "tears", "can't stop"]):
            if any(word in text_lower for word in ["no reason", "commercials", "everything"]):
                pregnancy_emotions["emotional_overwhelm"] = 0.7
        
        # Detect nesting instinct
        if any(word in text_lower for word in ["organize", "clean", "prepare", "nursery", "wash", "setup"]):
            pregnancy_emotions["nesting"] = 0.6
        
        # Determine primary emotion
        if pregnancy_emotions:
            primary_emotion = max(pregnancy_emotions, key=pregnancy_emotions.get)
            intensity = pregnancy_emotions[primary_emotion]
            contextual_triggers = list(pregnancy_emotions.keys())
        elif base_emotions:
            primary_emotion = max(base_emotions, key=base_emotions.get)
            intensity = base_emotions[primary_emotion]
            contextual_triggers = []
        else:
            primary_emotion = "neutral"
            intensity = 0.5
            contextual_triggers = []
        
        # Calculate trimester and factors
        if pregnancy_week <= 12:
            trimester = 1
            trimester_factor = 1.2  # More intense emotions
        elif pregnancy_week <= 27:
            trimester = 2
            trimester_factor = 0.9  # More stable
        else:
            trimester = 3
            trimester_factor = 1.1  # Building intensity
        
        # Calculate hormonal influence
        hormonal_influence = 0.7 if pregnancy_emotions else 0.3
        
        # Calculate confidence
        confidence = 0.8 if pregnancy_emotions else 0.6
        
        # Create result
        result = PregnancyEmotionalState(
            primary_emotion=primary_emotion,
            intensity=float(intensity),
            hormonal_influence=float(hormonal_influence),
            trimester_factor=float(trimester_factor),
            pregnancy_week=pregnancy_week,
            contextual_triggers=contextual_triggers,
            confidence_score=float(confidence)
        )
        
        return result
    
    def generate_supportive_response(self, emotional_state):
        """Generate supportive response based on emotional state"""
        
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        if emotion == "birth_anxiety" and intensity > 0.7:
            return "💝 Birth anxiety is so common, especially in the third trimester. Remember that your body is designed for this, and your healthcare team will support you every step of the way."
        
        elif emotion == "overwhelming_love":
            return "💕 The love you're feeling is one of the most beautiful parts of pregnancy. This deep connection with your baby is truly magical."
        
        elif emotion == "emotional_overwhelm":
            return "🌸 Pregnancy hormones can make everything feel so much more intense. It's completely normal to feel emotional - you're growing a life!"
        
        elif emotion == "nesting":
            return "🏠 That nesting instinct is in full swing! Your body is preparing you and your space for baby's arrival. Enjoy this burst of organizing energy!"
        
        else:
            return f"💝 I can sense you're feeling {emotion}. Whatever you're experiencing right now is completely valid and normal for your pregnancy journey."