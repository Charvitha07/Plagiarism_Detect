import re
from app.schemas.data_models import TextSegment
from app.preprocessing.cleaner import TextCleaner

class SentenceSegmenter:
    @staticmethod
    def segment(raw_pages: dict[int, str]) -> list[TextSegment]:
        segments = []
        sentence_idx = 0
        # Split on standard (.!?) and Indic (।॥) endpoints
        pattern = re.compile(r'[^.!?।॥\n]+[.!?।॥]*')
        
        for page_num, raw_text in raw_pages.items():
            for m in pattern.finditer(raw_text):
                span_text = m.group(0)
                
                # Compute non-whitespace exact bounds relative to original text array
                start_offset = len(span_text) - len(span_text.lstrip())
                end_offset = len(span_text.rstrip())
                
                actual_start = m.start() + start_offset
                actual_end = m.start() + end_offset
                
                orig_sentence = raw_text[actual_start:actual_end]
                clean_sentence = TextCleaner.clean(orig_sentence)
                
                if not clean_sentence:
                    continue
                    
                segments.append(TextSegment(
                    original_text=orig_sentence,
                    cleaned_text=clean_sentence,
                    start_char=actual_start,
                    end_char=actual_end,
                    page_number=page_num,
                    sentence_index=sentence_idx
                ))
                sentence_idx += 1
                
        return segments