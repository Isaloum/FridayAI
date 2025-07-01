from SelfQueryingCore import SelfQueryingCore

core = SelfQueryingCore()
results = core.suggest_followups("How can I make my AI smarter?")
print("Follow-up suggestions:")
for r in results:
    print(r)
