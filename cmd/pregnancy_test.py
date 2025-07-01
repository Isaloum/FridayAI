# ===================================================
# File: cmd/pregnancy_test.py
# Purpose: CLI testing for Pregnancy Domain
# ===================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force reloading in case the profile was stale/cached
import importlib
import core.pregnancy.PregnancyUserProfile
importlib.reload(core.pregnancy.PregnancyUserProfile)

from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.SelfNarrativeCore import log_event, update_mood
from core.pregnancy.PregnancyDomainMount import PregnancyDomainMount
from core.pregnancy.PregnancyPlanAdvisor import PregnancyPlanAdvisor
from core.pregnancy.PregnancyReflectionEngine import PregnancyReflectionEngine
from core.pregnancy.PregnancyUtils import parse_weeks_input
from word2number import w2n  # For handling "twenty-one" etc.

# Simulated identity object for testing/logging
class MockIdentity:
    def log_event(self, text, mood=None, source=None):
        print(f"[🧠 LOG] {text} (Mood: {mood}, Source: {source})")

def run_test():
    # Initialize memory, emotion, identity and mount the pregnancy domain
    memory = MemoryCore()
    emotion = EmotionCoreV2()
    identity = MockIdentity()
    pregnancy = PregnancyDomainMount(memory, emotion, identity)

    # Journey agent logs before week is set — may be 'None'
    print("\n📬 Journey Agent:\n" + pregnancy.journey.journey_update())
    print("\n📄 Blog Format:\n" + pregnancy.journey.for_blog())
    print("\n📧 Email Format:\n" + pregnancy.journey.for_email())
    print("\n👨‍👩‍👧 Family Update:\n" + pregnancy.journey.for_family())

    print("🤰 Pregnancy Domain Test")

    # Ask for week input and validate
    while True:
        raw_input_weeks = input("Weeks pregnant? (e.g. '22' or 'twenty-two')\n> ").strip().strip("'\"")
        weeks = parse_weeks_input(raw_input_weeks)
        if weeks is not None and 1 <= weeks <= 45:
            break
        print("❌ Invalid week number. Please enter a number between 1 and 45 (e.g. '22' or 'twenty-two').")

    # Save to profile and run updates
    pregnancy.profile.update_profile(weeks=weeks)
    print("\n📬 Journey Agent:\n" + pregnancy.journey.journey_update())

    # Trimester logic and pregnancy advice
    trimester = pregnancy.support.update_trimester(weeks)
    advisor = PregnancyPlanAdvisor(memory, trimester)

    # Suggest and log goals
    goals = advisor.suggest_goals()
    advisor.save_goal_suggestions(goals)

    print("\n🎯 Suggested Pregnancy Goals:")
    for g in goals:
        print(f"- {g}")

    # Repeat for emotional inputs
    while True:
        feeling = input("How are you feeling today?\n> ").strip()

        # 🎛 Tone toggle support
        if feeling.lower().startswith("tone:"):
            try:
                with open("last_reply.txt", "r", encoding="utf-8") as f:
                    last = f.read().strip()
                from core.ToneRewriterCore import ToneRewriterCore
                rewriter = ToneRewriterCore()
                tone = feeling.split(":", 1)[1].strip()
                print(f"\n🪄 Tone rewritten ({tone}):\n" + rewriter.rewrite(last, tone))
            except FileNotFoundError:
                print("⚠️ No reply to rewrite.")
            continue

        # 🔎 NLP Emotion Overlay (dev mode)
        analysis = pregnancy.support.tone_vectorizer.encode(feeling)
        print(f"[DEBUG] NLP Emotion: {analysis['primary']} ({analysis['certainty']*100:.1f}%)")

        # 🧠 Response and logging
        print("\n💬 Response to feeling:")
        response = pregnancy.support.respond_to_feeling(feeling)
        print(response)
        if "I'm not sure how to interpret that" in response:
            continue  # Skip journal + goal cycle if input is invalid


        # 💾 Store only meaningful responses
        if isinstance(response, str) and response.startswith("I'm not sure"):
            continue  # Skip saving nonsense or unclear inputs

        with open("last_reply.txt", "w", encoding="utf-8") as f:
            f.write(response)

        # 🎯 New goal set
        goals = advisor.suggest_goals()
        advisor.save_goal_suggestions(goals)

        print("\n🎯 Suggested Pregnancy Goals:")
        for g in goals:
            print(f"- {g}")

        # 📘 Journal log
        print("\n📘 Logging a journal entry:")
        pregnancy.journal.prompt_and_log()

        try:
            again = input("\nTest another entry? (y/n)\n> ")
        except EOFError:
            print("\n[🔚 Exiting Test Mode]")
            break
        if again.lower() != "y":
            break

    # End-of-session wrap-up
    engine = PregnancyReflectionEngine(memory)
    summary = engine.summarize_week()
    print("\n🪞 Weekly Reflection:\n" + summary)
    print("\n📊 Emotion Drift:\n" + pregnancy.emotion_drift.analyze_drift())
    print("\n🧬 Weekly Development Summary:")
    print(pregnancy.weekly_dev.get_update(pregnancy.profile.get_context().get("week", 0)))

if __name__ == "__main__":
    run_test()
