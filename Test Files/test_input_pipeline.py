from input_pipeline import InputSanitizer

sanitizer = InputSanitizer()

# Try a deliberate typo
raw = "definately"
result = sanitizer.sanitize(raw)

print("INPUT :", raw)
print("FIXED :", result)
