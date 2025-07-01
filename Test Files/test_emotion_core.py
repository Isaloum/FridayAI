
import pytest
from EmotionCore import EmotionCore

@pytest.fixture
def core():
    return EmotionCore()

def test_empty_input(core):
    result = core.analyze_detailed("")
    assert result["top_emotion"] == "neutral"
    assert result["confidence"] == 0.0

def test_single_emotion(core):
    result = core.analyze_detailed("I feel very sad and lonely today.")
    assert result["top_emotion"] == "sad"
    assert result["scores"]["sad"] > 0
    assert result["intensity"] >= 2

def test_mixed_emotions(core):
    result = core.analyze_detailed("I'm happy but also kind of anxious and stressed.")
    assert "happy" in result["scores"]
    assert "stressed" in result["scores"]
    assert result["top_emotion"] in result["scores"]

def test_mood_memory(core):
    core.analyze_detailed("I'm curious.")
    core.analyze_detailed("I'm stressed out.")
    state = core.get_mood_state()
    assert state["current_mood"] in ["curious", "stressed"]
    assert len(state["history"]) == 2

def test_emotional_response(core):
    message = core.get_emotional_response("love")
    assert "💖" in message
