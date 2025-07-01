from MemoryContextInjector import MemoryContextInjector

injector = MemoryContextInjector()
test_input = "How did I feel last time we talked about my sister?"
enriched = injector.inject_context(test_input)
print(enriched)
