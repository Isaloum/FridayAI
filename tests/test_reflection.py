# =============================================
# File: test_reflection.py
# Purpose: Unit test for SelfAwarenessEngine behavior
# =============================================

import unittest
from datetime import datetime, timedelta
from core.reflective_cognition.SelfAwarenessEngine import SelfAwarenessEngine

class MockMemoryCore:
    def query_memories(self, filter_tags=None, since_hours=24):
        return [
            {"content": "Reflected on task efficiency."},
            {"content": "Noticed stress handling improved."},
            {"content": "Logged emotional awareness after helping."}
        ]

class MockEmotionCore:
    def get_recent_mood_trend(self):
        return "Steady transition from anxious to calm."

class TestSelfAwarenessEngine(unittest.TestCase):
    def setUp(self):
        self.memory = MockMemoryCore()
        self.emotion = MockEmotionCore()
        self.engine = SelfAwarenessEngine(self.emotion, self.memory)

    def test_generate_self_reflection(self):
        reflection = self.engine.generate_self_reflection()
        self.assertIn("emotional state", reflection)
        self.assertIn("Reflected on task efficiency", reflection)
        self.assertIn("Steady transition", reflection)

    def test_should_trigger_drift_check(self):
        # First call should be true
        self.assertTrue(self.engine.should_trigger_drift_check())
        # Second call immediately after should be false
        self.assertFalse(self.engine.should_trigger_drift_check())

if __name__ == '__main__':
    unittest.main()
