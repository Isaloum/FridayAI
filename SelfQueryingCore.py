# SelfQueryingCore.py
# ------------------------
# Enables FridayAI to query LLMs for self-reflective responses.
# Integrates with Friday's memory to generate introspective, insightful replies.

import openai
import os
import logging

class SelfQueryingCore:
    def __init__(self, memory_core):
        """
        Initializes the SelfQueryingCore with access to MemoryCore and sets up logging.
        :param memory_core: instance of Friday's MemoryCore
        """
        self.memory = memory_core
        self.logger = self._init_logger()
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default fallback model

    def _init_logger(self):
        """
        Set up a dedicated logger for introspective queries.
        Logs all prompts and LLM replies for traceability.
        """
        logger = logging.getLogger("FridayAI.SelfQuerying")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler("self_query.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def ask_self(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM to generate a self-reflective response.
        :param prompt: A question or introspective input like "What emotions dominated last week?"
        :return: A thoughtful and emotionally-aware response string
        """
        system_prompt = "You are FridayAI, reflecting on a memory or emotion. Be brief, thoughtful, and insightful."

        # Attempt using OpenAI's newer SDK (openai>=1.0.0)
        try:
            if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content.strip()
                self.logger.info(f"[NewSDK] {prompt} => {reply}")
                return reply
        except Exception as e:
            self.logger.warning(f"New SDK failed: {e}")

        # Fallback to legacy SDK (openai<1.0.0)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            reply = response["choices"][0]["message"]["content"].strip()
            self.logger.info(f"[LegacySDK] {prompt} => {reply}")
            return reply
        except Exception as e:
            self.logger.error(f"All SDK calls failed: {e}")
            return "[Error: LLM unavailable] — fallback: I'm here to help, even if I can't always reach my deeper thoughts."
