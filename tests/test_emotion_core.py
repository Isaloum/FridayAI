# =============================================
# File: test_emotion_core.py
# Purpose: Unit tests for EmotionCoreV2 mood tracking
# =============================================

import unittest
from datetime import datetime, timedelta
from core.EmotionCoreV2 import EmotionCoreV2

class TestEmotionCoreV2(unittest.TestCase):
    def setUp(self):
        self.core = EmotionCoreV2()

    def test_adjust_mood_positive(self):
        base = self.core.mood
        self.core.adjust_mood(0.2)
        self.assertGreater(self.core.mood, base)

    def test_adjust_mood_negative(self):
        base = self.core.mood
        self.core.adjust_mood(-0.2)
        self.assertLess(self.core.mood, base)

    def test_mood_bounds(self):
        self.core.adjust_mood(100)
        self.assertLessEqual(self.core.mood, 1.0)
        self.core.adjust_mood(-100)
        self.assertGreaterEqual(self.core.mood, -1.0)

    def test_get_mood_history_format(self):
        history = self.core.get_mood_history(since_hours=1)
        self.assertIsInstance(history, list)

if __name__ == '__main__':
    unittest.main()
