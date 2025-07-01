
import pytest
from VectorMemoryCore import VectorMemoryCore

@pytest.fixture
def vm():
    return VectorMemoryCore()

def test_ingest_and_query_exact_match(vm):
    vm.ingest("The mitochondria is the powerhouse of the cell.")
    results = vm.query("What is the powerhouse of the cell?")
    assert len(results) > 0
    assert "mitochondria" in results[0]["text"].lower()

def test_query_no_data():
    vm = VectorMemoryCore()
    results = vm.query("Anything here?")
    assert results == []

def test_top_k_limit(vm):
    texts = [
        "AI learns through data.",
        "Humans learn through experience.",
        "Cats are curious animals.",
        "Python is a programming language.",
        "FridayAI is an emotional assistant."
    ]
    for t in texts:
        vm.ingest(t)
    results = vm.query("Tell me about learning.", top_k=2)
    assert len(results) == 2
    assert any("learn" in r["text"].lower() for r in results)
