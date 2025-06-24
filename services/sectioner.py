import logging
from transformers import pipeline


logger = logging.getLogger(__name__)

SECTION_LABELS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references"
]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_section_title(line: str):
    """Classify a section title into a standard section label."""
    result = classifier(line, SECTION_LABELS)
    return result["labels"][0], result["scores"][0]

def split_into_sections(pages: list, threshold: float = 0.6):
    """Split cleaned pages into semantically classified sections."""
    sections = {label: "" for label in SECTION_LABELS}
    sections["preamble"] = ""
    current_section = "preamble"
    
    for page in pages:
        for line in page["text"].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Attempt to classify every line sematically
            predicted_label, confidence= classify_section_title(stripped)
            # Recognize header if confidence is high enough
            if confidence >= threshold:
                # Start a new section when a header is detected
                if predicted_label in SECTION_LABELS:
                    current_section = predicted_label
                    logger.info(f"Switched to section: {current_section}")
                continue
            # Otherwise, append content to the current section
            sections[current_section] += stripped + "\n"
    
    # If no sections detected beyond preamble, log warning and return full_text under "full_text"
    non_empty = {k: v for k, v in sections.items() if v and k != "preamble"}
    if not non_empty:
        full_text = "\n".join(p["text"] for p in pages)
        logger.warning("No sections detected semantically; returning full_text.")
        return {"full_text": full_text}
    
    return sections