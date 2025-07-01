# =====================================
# File: FridayAI.py (Legacy Preserved)
# Purpose: Full Cognitive Core with Legacy Blocks Retained
# =====================================

# (Header, imports, env vars – [unchanged from optimized version])

# ... [CODE OMITTED HERE FOR SPACE. Will regenerate and upload full file] ...

# === Legacy: Old respond_to logic (pre-pipeline) ===
# cleaned = self.input_sanitizer.sanitize(user_input)
# intent_result = self.router.route(cleaned)
# domain = intent_result.get("domain", "default_chat")
# tone = intent_result.get("emotional_tone", "neutral")
# confidence = intent_result.get("confidence", 0.5)
#
# context = self.context_injector.inject(cleaned)
# reflection = context.get("reflection", "")
# snippets = context.get("memory_snippets", [])
# if not isinstance(snippets, list):
#     snippets = []
#
# emotions = self.emotion_layer.detect_emotion_layers(cleaned)
# mood_memories = self.mood_filter.recall_by_mood(user_input)
# for m in mood_memories['memories']:
#     print(f"[Mood-Matched Memory] {m['text']} (score={m['score']:.2f})")
# emotion_summary = emotions.get("primary", "neutral")

# empathy_result = self.empathy.analyze_subtext(cleaned)
# inferred = empathy_result.get("inferred", {})
# strategy = empathy_result.get("strategy", {})
# self.anchor_logger.log_anchor_if_deep("user", cleaned, inferred, strategy)

# empathy_cue = empathy_result.get("empathy_cue", "")
# empathy_line = empathy_result.get("empathy_reply", "")

# tone_profile = self.personality.get_profile()
# tone_profile.update(self.tone_rebalancer.analyze_user_style(cleaned))
# tone_profile = self.emotional_anchors.apply_anchors_to_tone(tone_profile)

# vector_hits = self.vector_memory.query(cleaned, top_k=2, domain=domain)
# vector_summary = MemorySummarizer.summarize_vector_hits(vector_hits)

# if vector_hits:
#     print(f"\n🧠 Memory Insight: {vector_summary}")

# self.belief_updater.ingest_reflection(
#     summary=vector_summary,
#     context={
#         "emotion": emotion_summary,
#         "intent": domain,
#         "source": "vector_memory",
#         "time": datetime.now().isoformat()
#     }
# )

# try:
#     from LLMCore import LLMCore
#     raw_reply = self.llm.prompt(cleaned)
#     print(f"\n[DEBUG] Friday REPLY RAW: {raw_reply}")
# except Exception:
#     raw_reply = "Sorry, my reasoning engine hit an error."

# reply = raw_reply.strip()
# if not reply:
#     reply = "[ERROR] LLM returned empty or null response."

# self.drift_sim.simulate_drift(user_input)
# emotional_weight = estimate_emotional_weight(cleaned, emotion_summary)

# self.auto_learner.learn_from_input_output(cleaned, reply, {
#     "emotion": emotion_summary,
#     "timestamp": datetime.now().isoformat()
# })

# self.narrative_fusion.log_event(user_input, emotion_summary, source="user")
# self.narrative_fusion.log_event(reply, emotion_summary, source="friday")

# self.identity.log_event(user_input, kind="event")
# self.identity.update_mood(emotion_summary)

# store_memory("user", {
#     "input": user_input,
#     "reply": reply,
#     "emotion": emotion_summary,
#     "intent": domain,
#     "time": datetime.now().isoformat(),
#     "emotional_weight": emotional_weight
# })

# vector_hits = self.vector_memory.query(
#     prompt=cleaned,
#     top_k=2,
#     domain=domain,
#     mood=emotion_summary
# )

# if vector_hits:
#     print("\n[Vector Recall - Mood & Domain Matched]")
#     for mem in vector_hits:
#         print(f"- {mem['text']} (score={mem['score']:.2f})")

# similar_memories = self.vector_memory.query(user_input, top_k=2, domain=domain)
# if similar_memories:
#     print("\n[Vector Recall] Related:")
#     for mem in similar_memories:
#         print(f"- {mem['text']} (score={mem['score']:.2f})")

# return {
#     "domain": domain,
#     "content": reply,
#     "confidence": confidence,
#     "emotional_tone": emotion_summary,
#     "processing_time": datetime.now().isoformat()
# }
