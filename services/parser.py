import pdfplumber
import logging
from fastapi import UploadFile
from collections import Counter

type_logging = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_page_text(text, common_lines=None):
    """Remove empty lines and optionally strip common headers/footers."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if common_lines:
        lines = [line for line in lines if line not in common_lines]
    return "\n".join(lines)


def find_common_headers_footers(pages):
    """Find lines that appear consistently at the top/bottom of each page"""
    headers = Counter()
    footers = Counter()
    for page in pages:
        lines = page["text"].splitlines()
        if lines:
            headers[lines[0].strip()] += 1
            footers[lines[-1].strip()] += 1
    total = len(pages)
    header_set = {line for line, count in headers.items() if count > total * 0.6}
    footer_set = {line for line, count in footers.items() if count > total * 0.6}
    type_logging.info(f"Detected {len(header_set)} common headers and {len(footer_set)} common footers")
    return {"headers": header_set, "footers": footer_set}

def extract_pdf(file: UploadFile):
    """Extract text from each page of a PDF uploaded via FastAPI's UploadFile."""
    
    raw_pages = []
    cleaned_pages = []
    
    try:
        with pdfplumber.open(file.file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    raw_pages.append({
                        "page": i, 
                        "text": text.strip()
                    })
        type_logging.info(f"Extracted raw text from {len(raw_pages)} pages")
        
        # Find and remove headers/footers
        common = find_common_headers_footers(raw_pages)                  
        
        # Clean text per page
        for page in raw_pages:
            cleaned = clean_page_text(page["text"], common_lines=common["headers"] | common["footers"])
            cleaned_pages.append({"page": page["page"], "text": cleaned})
            
        type_logging.info(f"Cleaned text for {len(cleaned_pages)} pages")
    
    except Exception as err:
        type_logging.error("Error parsing PDF", exc_info=True)
        raise RuntimeError(f"Failed to parse PDF: {err}")
    finally:
        # Reset stream cursor for potential re-use
        file.file.seek(0)
    full_text = "\n\n".join(p["text"] for p in cleaned_pages)
    return {"pages": cleaned_pages, "full_text": full_text}