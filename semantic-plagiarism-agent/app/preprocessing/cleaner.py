import re
import unicodedata

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        # Normalize Unicode (e.g., combining characters)
        text = unicodedata.normalize('NFKC', text)
        # Remove control characters except standard whitespace
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\t', '\r'])
        # Normalize continuous whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()