# FridayAI_AutoTest.py

import time
from MemoryCore import MemoryCore
from FridayAI import FridayAI
from FactExpiration import FactExpiration
from ErrorRecovery import ErrorRecovery

# Initialize components
memory = MemoryCore()
friday = FridayAI(memory)
expiration = FactExpiration(expiration_time_seconds=5)  # Short for test
recovery = ErrorRecovery()

# Utility function
def test_case(name, func, expected):
    result = func()
    if result == expected:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} → Got: '{result}', Expected: '{expected}'")

# 1. Add facts
memory.add_fact("favorite_car", "Shelby")
memory.add_fact("location", "Laval, QC")

# 2. Test simple fact retrieval
test_case(
    "Favorite Car Retrieval",
    lambda: friday.respond_to("What is my favorite car?"),
    "Your favorite car is Shelby."
)

test_case(
    "Location Retrieval",
    lambda: friday.respond_to("Where am I?"),
    "You are in Laval, QC."
)

# 3. Test unknown fact recovery
test_case(
    "Unknown Fact Recovery",
    lambda: friday.respond_to("What is my favorite color?"),
    "Fact 'favorite color' not found. Please update or specify."
)

# 4. Test fact expiration
print("[INFO] Waiting 6 seconds for expiration test...")
time.sleep(6)

# Clean expired facts
expiration.remove_expired_facts(memory.memory)

test_case(
    "Expired Favorite Car Fact",
    lambda: friday.respond_to("What is my favorite car?"),
    "Fact 'favorite car' not found. Please update or specify."
)

# 5. Test chain command simulation
def chain_commands():
    responses = []
    responses.append(friday.respond_to("Where am I?"))
    responses.append(friday.respond_to("What is my favorite car?"))
    responses.append(friday.respond_to("System status?"))
    return " | ".join(responses)

test_case(
    "Chain Command Handling",
    chain_commands,
    "You are in Laval, QC. | Fact 'favorite car' not found. Please update or specify. | Systems operational."
)

print("\n✅ Auto Test Script Completed.")
