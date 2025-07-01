# ==============================================
# File: InputSanitizer.py
# Purpose: Clean user text without heavy models
# Author: Fixed version - no external models needed
# ==============================================

import re

class InputSanitizer:
    """
    Simple text cleaner that works without downloading big files.
    Fixes basic typos and formatting issues.
    """
    
    def __init__(self, symspell_dict_path=None, jamspell_model_path=None):
        """
        Start the text cleaner.
        We ignore the file paths - we don't need them anymore.
        """
       #print("[InputSanitizer] Ready! Using simple mode (no big files needed)")
        
    def ekphrasis_clean(self, text):
        """
        Fix basic text problems like too many !!! or spaces.
        """
        # Fix too many punctuation marks
        text = re.sub(r'[!]{2,}', '!', text)  # !!! becomes !
        text = re.sub(r'[?]{2,}', '?', text)  # ??? becomes ?
        text = re.sub(r'[.]{3,}', '...', text)  # ..... becomes ...
        
        # Fix extra spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces become one
        
        # Turn simple emojis into words
        text = text.replace(':)', ' happy ')
        text = text.replace(':(', ' sad ')
        text = text.replace(':D', ' very happy ')
        
        return text.strip()

    def symspell_correct(self, text):
        """
        Fix common spelling mistakes.
        """
        # List of common typos and their fixes
        fixes = {
            'teh': 'the',
            'recieve': 'receive', 
            'definately': 'definitely',
            'occured': 'occurred',
            'seperately': 'separately',
            'begining': 'beginning',
            'untill': 'until',
            'wich': 'which',
            'youre': 'you are',
            'ur': 'you are'
        }
        
        words = text.split()
        fixed_words = []
        
        for word in words:
            # Get the word without punctuation
            clean_word = word.lower().strip('.,!?')
            
            # Check if we need to fix it
            if clean_word in fixes:
                # Keep the original capitalization
                if word[0].isupper():
                    fixed_words.append(fixes[clean_word].capitalize())
                else:
                    fixed_words.append(fixes[clean_word])
            else:
                # Keep the original word
                fixed_words.append(word)
                
        return ' '.join(fixed_words)

    def jamspell_fix(self, text):
        """
        Fix basic grammar problems.
        """
        # Fix common grammar issues
        text = re.sub(r'\bi\b', 'I', text)  # i becomes I
        text = re.sub(r'\bim\b', "I'm", text, flags=re.IGNORECASE)  # im becomes I'm
        text = re.sub(r'\bdont\b', "don't", text, flags=re.IGNORECASE)  # dont becomes don't
        text = re.sub(r'\bcant\b', "can't", text, flags=re.IGNORECASE)  # cant becomes can't
        text = re.sub(r'\bwont\b', "won't", text, flags=re.IGNORECASE)  # wont becomes won't
        
        return text

    def sanitize(self, raw_input: str) -> str:
        """
        Main function: Clean the user's text in 3 steps.
        Step 1: Fix formatting
        Step 2: Fix spelling  
        Step 3: Fix grammar
        """
        # Safety check - make sure we have text to work with
        if not raw_input or not isinstance(raw_input, str):
            return ""
            
        # Clean the text in 3 steps
        step1 = self.ekphrasis_clean(raw_input)
        step2 = self.symspell_correct(step1)
        step3 = self.jamspell_fix(step2)
        
        return step3