# ==============================================
# File: core/HumanConversationCore.py
# Purpose: Natural human-like dialogue engine using DialoGPT
# ==============================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HumanConversationCore:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    def reply(self, user_input: str) -> str:
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = (
            torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
            if self.chat_history_ids is not None
            else new_input_ids
        )

        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response

    def reset(self):
        self.chat_history_ids = None
