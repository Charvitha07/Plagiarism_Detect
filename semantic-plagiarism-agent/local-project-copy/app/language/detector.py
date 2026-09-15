import logging
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

class LanguageDetector:
    @staticmethod
    def detect(text: str) -> tuple[str, float]:
        if not text or len(text.strip()) < 3:
            return "unknown", 0.0
            
        try:
            langs = detect_langs(text)
            if langs:
                return langs[0].lang, langs[0].prob
        except LangDetectException as e:
            logger.debug(f"Language detection failed (likely short/symbolic text): {e}")
            
        return "unknown", 0.0