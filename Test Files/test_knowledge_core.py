
import pytest
from KnowledgeCore import KnowledgeCore

@pytest.fixture
def core():
    return KnowledgeCore()

def test_yul_transport_match(core):
    query = "How do I get from Laval to YUL airport by bus or Uber?"
    result = core.lookup(query)
    assert "YUL" in result

def test_emergency_number_match(core):
    query = "What's the emergency number in Canada if I need help?"
    result = core.lookup(query)
    assert "911" in result

def test_no_match(core):
    query = "Tell me about ancient Roman pottery techniques."
    result = core.lookup(query)
    assert result is None
