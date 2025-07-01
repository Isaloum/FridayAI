# =============================================
# File: SelfIntentModel.py
# Purpose: Classify whether user input refers to Friday's internal state
# Labels: self_mood, self_memory, self_identity, self_behavior, user_emotion, other
# =============================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SelfIntentModel:
    """
    Uses a transformer to classify self-referential intent in user input.
    """

    def __init__(self, model_name="distilbert-base-uncased", label_map=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.label_map = label_map or {
            0: "self_mood",
            1: "self_memory",
            2: "self_identity",
            3: "self_behavior",
            4: "user_emotion",
            5: "other"
        }

    def predict_intent(self, text: str):
        """
        Classify intent and return:
        {
            "label": str,
            "confidence": float
        }
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            top_class = torch.argmax(probs, dim=1).item()
            confidence = round(probs[0][top_class].item(), 3)

        return {
            "label": self.label_map[top_class],
            "confidence": confidence
        }
