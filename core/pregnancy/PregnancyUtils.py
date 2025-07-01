# ===================================================
# File: core/pregnancy/PregnancyUtils.py
# Purpose: Common utilities for pregnancy domain
# ===================================================

from word2number import w2n

VALID_FEELINGS = [
    "happy", "sad", "anxious", "tired", "angry", "excited", "calm", "worried",
    "neutral", "great", "good", "not so good", "not very well", "can't tell"
]

def is_valid_feeling(text: str) -> bool:
    text = text.lower()
    return any(word in text for word in VALID_FEELINGS)

def parse_weeks_input(raw_input: str) -> int | None:
    raw_input = raw_input.strip().lower()
    try:
        return w2n.word_to_num(raw_input)
    except:
        pass
    try:
        return int(raw_input)
    except:
        return None
