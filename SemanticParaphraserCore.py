# ======================================
# File: SemanticParaphraserCore.py
# Purpose: Enables FridayAI to semantically reframe input and internal thoughts with clarity and tone alignment
# ======================================

import re
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Optional: use Friday's trait system and memory injection logic
# from memory_engine import MemoryInjector
# from emotion_engine import EmotionTagger
# from trait_blender import TraitBlender

class SemanticParaphraserCore:
    # Semantic cortex to paraphrase inputs with emotional and contextual awareness
    def __init__(self, model_name="t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Placeholder for emotion and trait systems
        self.trait_profile = {
            "humor": 0.2,
            "warmth": 0.7,
            "precision": 0.6
        }

    def _prepare_input(self, text: str) -> str:
        # Prepends paraphrasing instruction for T5 model
        return f"paraphrase: {text.strip()}"

    def _post_process(self, text: str) -> str:
        # Basic cleanup and formatting of model output
        text = re.sub(r"\s+", " ", text).strip()
        text = text[0].upper() + text[1:] if text else text
        return text

    def paraphrase(self, raw_text: str, context: Dict[str, Any] = None) -> str:
        # Runs semantic rewriting pipeline
        input_text = self._prepare_input(raw_text)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(self.device)

        outputs = self.model.generate(
            inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        rewritten = self._post_process(decoded)

        # Inject empathy, context, emotion shaping here (future hook)
        return rewritten


# === EXAMPLE USAGE ===
# Demo: transforms raw emotional street slang to a clean, emotionally aligned sentence
if __name__ == "__main__":
    paraphraser = SemanticParaphraserCore()
    raw = "yo bro i’m deadass i can’t even wit her rn fr she wildin"
    clean = paraphraser.paraphrase(raw)
    print("REWRITE:", clean)

    # OUTPUT (approx): "I’m completely serious, I just can’t handle her right now—she’s acting wild."
