# ==============================================
# File: C:\Users\ihabs\FridayAI\tests\test_api_service.py
# Purpose: Pytest unit tests for Friday AI API service endpoints
# ==============================================

import sys, os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from api_service import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    # Stub AI methods for testing
    import api_service
    api_service.ai.respond_to = lambda text: { 'content': 'dummy response', 'emotional_tone': 'neutral', 'suggestions': [] }
    class DummyReflectionLoop:
        def run_reflection_cycle(self, **kwargs):
            return { 'insights': [], 'belief_updates': 0 }
    api_service.ai.reflection_loop = DummyReflectionLoop()
    with flask_app.test_client() as client:
        yield client


def test_respond_success(client):
    payload = {"input": "Hello Friday AI!"}
    resp = client.post('/api/v1/respond', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert "emotional_tone" in data
    assert isinstance(data.get("suggestions"), list)
    assert "metadata" in data
    assert "request_id" in data["metadata"]


def test_respond_empty_input(client):
    payload = {"input": ""}
    resp = client.post('/api/v1/respond', json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data.get("error") == "Input cannot be empty"


def test_reflect_success(client):
    resp = client.post('/api/v1/reflect', json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("status") == "success"
    assert isinstance(data.get("insights"), list)
    assert "belief_updates" in data
