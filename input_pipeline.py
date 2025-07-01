# ========================================
# File: input_pipeline.py – Cleaned + Neural
# Purpose: Modular input sanitizer for FridayAI
# Includes Ekphrasis, SymSpell, and NeuralNormalizerCore
# ========================================

import os
import requests
from symspellpy import SymSpell, Verbosity
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer

from NeuralNormalizerCore import NeuralNormalizerCore

DEBUG_MODE = os.getenv("FRIDAY_DEBUG", "false").lower() == "true"

ASSETS_DIR = os.path.join(os.getcwd(), "assets")
SYM_DICT_URL = "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt"
SYM_DICT_FILE = os.path.join(ASSETS_DIR, "frequency_dictionary_en_82_765.txt")


def download_file(url, path):
    print(f"[INFO] Downloading: {os.path.basename(path)}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print(f"[✓] Downloaded: {path}")
    else:
        raise Exception(f"Failed to download {url} — HTTP {r.status_code}")


def ensure_assets():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(SYM_DICT_FILE):
        download_file(SYM_DICT_URL, SYM_DICT_FILE)


class InputSanitizer:
    def __init__(self):
        ensure_assets()

        # 1. SymSpell Setup
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.sym_spell.load_dictionary(SYM_DICT_FILE, term_index=0, count_index=1)

        # Load custom corrections (slang, broken words, etc.)
        custom_dict = os.path.join(ASSETS_DIR, "custom_words.txt")
        if os.path.exists(custom_dict):
            self.sym_spell.load_dictionary(custom_dict, term_index=0, count_index=1)

        # 2. Ekphrasis Setup
        self.ekphrasis = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time',
                       'date', 'number'],
            annotate={},
            fix_html=True,
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=True,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            emoticons=emoticons
        )

        # 3. Neural Rewriter (Flan-T5 small)
        self.neural_normalizer = NeuralNormalizerCore()

    def ekphrasis_clean(self, text):
        return " ".join(self.ekphrasis.pre_process_doc(text))

    def symspell_correct(self, text):
        corrected_words = []
        missed_words = []

        for word in text.split():
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
                missed_words.append(word)

        # Log missed words to help expand custom_words.txt later
        if missed_words:
            with open(os.path.join(ASSETS_DIR, "missed_words.log"), "a", encoding="utf-8") as log_file:
                for word in missed_words:
                    log_file.write(word + "\n")

        return " ".join(corrected_words)

    def sanitize(self, raw_input):
        if DEBUG_MODE:
            print("[DEBUG] Raw Input:", raw_input)
        step1 = self.ekphrasis_clean(raw_input)
        if DEBUG_MODE:
            print("[DEBUG] After Ekphrasis:", step1)
        step2 = self.symspell_correct(step1)
        if DEBUG_MODE:
            print("[DEBUG] After SymSpell:", step2)

        # Slang/alias fallback (quick patch before Neural)
        step2 = step2.replace(" u ", " you ").replace(" im ", " i am ").replace(" wht ", " what ")
        step2 = step2.replace(" r ", " are ").replace(" ya ", " you ").replace(" cant ", " can't ")

        step3 = self.neural_normalizer.normalize(step2)
        if DEBUG_MODE:
            print("[DEBUG] After Neural Normalizer:", step3)
        return step3

