import logging
from typing import List, Dict
from langdetect import detect, LangDetectException
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cache translator pipelines per source language
_translators = {}

# Supported source languages for translation
SUPPORTED_SRC_LANGS = [
    "af", "sq", "am", "ar", "hy", "az", "bn", "bs", "bg", "ca",
    "zh", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "gl",
    "ka", "de", "el", "gu", "ha", "he", "hi", "hu", "is", "id",
    "it", "ja", "jv", "kn", "kk", "km", "ko", "lo", "lv", "lt",
    "mk", "ms", "ml", "mr", "ne", "ny", "pa", "fa", "pl", "pt",
    "ro", "ru", "sr", "si", "sk", "sl", "es", "sw", "sv", "ta",
    "te", "th", "tr", "uk", "ur", "vi", "cy", "yo", "zu"
]


def detect_language(text: str):
    """Detect the language of a given text snippet."""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        logger.warning("Language detection failed, defaulting to 'unknown'.")
        return 'unknown'


def get_translator(src_lang: str):
    """Return a translation pipeline that translates from src_lang to English."""
    if src_lang not in _translators:
        if src_lang not in SUPPORTED_SRC_LANGS:
            raise ValueError(f"Unsupported source language: {src_lang}")
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        # Initialize the translation pipeline
        _translators[src_lang] = pipeline("translation", model=model_name)
        logger.info(f"Loaded translator for language: {src_lang}")
    return _translators[src_lang]


def translate_text(text: str, src_lang: str):
    """Translate text from src_lang to English."""
    try:
        translator = get_translator(src_lang)
        result = translator(text)
        return result[0]['translation_text'] # Return translated text
    except Exception as e:
        logger.error(f"Translation error for lang={src_lang}: {e}")
        return text  # Return original


def ensure_english_pages(pages: List[Dict], sample_ratio: float = 0.2):
    """Detect if pages are non-English and translate all page texts to English."""
    # Combine a sample of page texts for detection
    sample_count = max(1, int(len(pages) * sample_ratio))
    sample_text = "\n".join(pages[i]['text'] for i in range(sample_count))
    lang = detect_language(sample_text)

    if lang == 'en' or lang == 'unknown':
        return pages

    logger.info(f"Detected language '{lang}', translating {len(pages)} pages to English.")
    try:
        translator = get_translator(lang)
    except ValueError:
        # Unsupported language; log and skip translation
        logger.warning(f"No translator available for language '{lang}', skipping translation.")
        return pages

    # Batch translation for performance: feed list of texts to the pipeline
    texts = [page['text'] for page in pages]
    translations = translator(texts)
    # Reconstruct pages with translated text
    translated_pages = []
    for idx, page in enumerate(pages):
        translated_text = translations[idx].get('translation_text', page['text'])
        translated_pages.append({'page': page['page'], 'text': translated_text})

    return translated_pages