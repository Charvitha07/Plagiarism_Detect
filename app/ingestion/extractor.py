import fitz  # PyMuPDF
import docx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExtractionError(Exception): pass

class DocumentExtractor:
    @staticmethod
    def extract(file_path: str) -> dict[int, str]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                return DocumentExtractor._extract_pdf(path)
            elif ext == ".docx":
                return DocumentExtractor._extract_docx(path)
            elif ext in [".txt", ".md"]:
                return DocumentExtractor._extract_txt(path)
            else:
                raise ExtractionError(f"Unsupported format: {ext}")
        except Exception as e:
            logger.error(f"Error extracting {file_path}: {str(e)}")
            raise ExtractionError(f"Failed to process {path.name}: {str(e)}")

    @staticmethod
    def _extract_pdf(path: Path) -> dict[int, str]:
        pages = {}
        with fitz.open(path) as doc:
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    pages[page_num + 1] = text
        return pages

    @staticmethod
    def _extract_docx(path: Path) -> dict[int, str]:
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return {1: text}

    @staticmethod
    def _extract_txt(path: Path) -> dict[int, str]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {1: text}