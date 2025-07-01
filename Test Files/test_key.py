import openai

openai.api_key = "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"

try:
    models = openai.Model.list()
    print("✅ API Key is working. Models available:")
    for model in models["data"]:
        print("-", model["id"])
except Exception as e:
    print("❌ Error:", e)
