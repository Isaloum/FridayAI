# ==========================================
# File: NeuralNormalizerCore.py
# Purpose: Neural-powered correction & paraphrasing for FridayAI input
# Model: Flan-T5-small (offline-compatible, general-purpose)
# ==========================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class NeuralNormalizerCore:
    """
    Uses a transformer model (Flan-T5-small) to correct grammar,
    expand slang, and rewrite informal/messy input into canonical form.
    """

    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def normalize(self, text: str) -> str:
        """
        Rewrites input text using T5 prompt to clean grammar, slang, and structure.
        """
        prompt = f"paraphrase and correct: {text.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output


# =============================
# CLI Test Mode (Offline)
# =============================
if __name__ == "__main__":
    nn = NeuralNormalizerCore()
    print("\n🧠 NeuralNormalizerCore (Flan-T5) Ready")
    while True:
        try:
            raw = input("You (raw): ").strip()
            if raw.lower() in ["exit", "quit"]:
                break
            clean = nn.normalize(raw)
            print(f"\n✅ Normalized: {clean}\n")
        except KeyboardInterrupt:
            break
