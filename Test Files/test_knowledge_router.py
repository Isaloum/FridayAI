
import pytest
from KnowledgeRouter import KnowledgeRouter

@pytest.fixture
def router():
    return KnowledgeRouter()

def test_detect_domain_transport(router):
    query = "How do I get a taxi to the airport?"
    domain = router.detect_domain(query)
    assert domain == "transport"

def test_detect_domain_emergency(router):
    query = "What number do I call in an emergency?"
    domain = router.detect_domain(query)
    assert domain == "emergency"

def test_unknown_domain(router):
    query = "What's the recipe for banana bread?"
    domain = router.detect_domain(query)
    assert domain == "general"

def test_add_and_lookup_fact(router):
    router.add_fact(
        label="ai_reboot_instructions",
        response="To reboot the AI, press Ctrl+C then rerun the script.",
        keywords=["ai", "reboot", "restart", "friday"]
    )
    result = router.handle_query("How do I reboot Friday?")
    assert result["domain"] == "knowledge"
    assert "Ctrl+C" in result["content"]

def test_memory_fallback(router):
    router.memory.save_fact("user.favorite_color", "blue", source="test")
    result = router.handle_query("user.favorite_color")
    assert result["domain"] == "memory"
    assert "blue" in result["content"]
