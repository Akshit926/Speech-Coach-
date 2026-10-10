"""
Language Detection Module for ToastMasterAI
Detects the language of the speech transcript.
"""

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def detect_language(text: str) -> dict:
    """
    Detect the language of the input text.
    
    Returns:
        Dictionary with 'language', 'confidence', 'is_english', 'message'
    """
    if not text or len(text.strip()) < 10:
        return {
            'language': 'Unknown',
            'language_code': 'unknown',
            'confidence': 0.0,
            'is_english': False,
            'message': 'Text is too short for reliable language detection.'
        }
    
    if not LANGDETECT_AVAILABLE:
        # Fallback: simple heuristic
        return _heuristic_detection(text)
    
    try:
        lang_code = detect(text)
        lang_probs = detect_langs(text)
        
        confidence = 0.0
        for lp in lang_probs:
            if str(lp).startswith(lang_code):
                confidence = lp.prob
                break
        
        lang_name = LANGUAGE_MAP.get(lang_code, lang_code.upper())
        is_english = lang_code == 'en'
        
        if is_english:
            message = f"Language: {lang_name}"
        else:
            message = (
                f"Detected Language: {lang_name}. "
                "This system currently provides detailed analysis primarily for English speeches."
            )
        
        return {
            'language': lang_name,
            'language_code': lang_code,
            'confidence': round(confidence * 100, 1),
            'is_english': is_english,
            'message': message
        }
    except Exception as e:
        return {
            'language': 'Unknown',
            'language_code': 'unknown',
            'confidence': 0.0,
            'is_english': False,
            'message': f'Language detection failed: {str(e)}'
        }


def _heuristic_detection(text: str) -> dict:
    """Simple English detection fallback when langdetect is unavailable."""
    common_english = {'the', 'is', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'was', 'for', 'i'}
    words = text.lower().split()
    if not words:
        return {
            'language': 'Unknown', 'language_code': 'unknown',
            'confidence': 0.0, 'is_english': False,
            'message': 'Could not detect language.'
        }
    
    english_count = sum(1 for w in words if w in common_english)
    ratio = english_count / len(words)
    
    is_english = ratio > 0.15
    return {
        'language': 'English' if is_english else 'Non-English',
        'language_code': 'en' if is_english else 'unknown',
        'confidence': round(ratio * 100, 1),
        'is_english': is_english,
        'message': 'Language: English' if is_english else 'This system provides detailed analysis primarily for English speeches.'
    }


# Language code to name mapping
LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'mr': 'Marathi', 'ta': 'Tamil',
    'te': 'Telugu', 'bn': 'Bengali', 'gu': 'Gujarati', 'kn': 'Kannada',
    'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Urdu',
    'fr': 'French', 'de': 'German', 'es': 'Spanish', 'pt': 'Portuguese',
    'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'zh-cn': 'Chinese',
    'ar': 'Arabic', 'ru': 'Russian', 'nl': 'Dutch', 'sv': 'Swedish',
}
