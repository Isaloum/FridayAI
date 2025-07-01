from MemoryContextInjector import MemoryContextInjector

injector = MemoryContextInjector()

test_input = "How did I feel last time we talked about my sister?"

# Clean output (default)
clean = injector.inject_context(test_input, show_metadata=False)
print("\n🧼 Clean Output:")
print(clean)

# Verbose output (debug/info mode)
verbose = injector.inject_context(test_input, show_metadata=True)
print("\n🔍 Verbose Output:")
print(verbose)
