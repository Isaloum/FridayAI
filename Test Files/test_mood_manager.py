
import pytest
from MoodManagerCore import MoodManagerCore
import os

@pytest.fixture
def mood_manager(tmp_path):
    file_path = tmp_path / "test_mood.json"
    return MoodManagerCore(state_file=str(file_path))

def test_update_and_get_current_mood(mood_manager):
    mood_manager.update_mood("session1", "sad")
    mood_manager.update_mood("session1", "happy")
    mood_manager.update_mood("session1", "sad")

    current = mood_manager.get_current_session_mood("session1")
    assert current == "sad"

def test_mood_report(mood_manager):
    mood_manager.update_mood("sessionA", "angry")
    mood_manager.update_mood("sessionA", "angry")
    mood_manager.update_mood("sessionB", "happy")

    report = mood_manager.get_mood_report()
    assert report['dominant_mood'] == "angry"
    assert report['distribution']["angry"] == 2
    assert report['distribution']["happy"] == 1

def test_reset_mood(mood_manager):
    mood_manager.update_mood("sX", "love")
    mood_manager.reset()
    report = mood_manager.get_mood_report()
    assert report['dominant_mood'] == "neutral"
    assert report['distribution'] == {}
