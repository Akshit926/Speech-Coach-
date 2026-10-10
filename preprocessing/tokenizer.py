"""
Tokenization Module for ToastMasterAI
Handles word tokenization and sentence segmentation using NLTK and spaCy.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def tokenize_words(text: str) -> list:
    """
    Tokenize text into individual words.
    
    Args:
        text: Input text string
    
    Returns:
        List of word tokens
    """
    if not text or not isinstance(text, str):
        return []
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback to simple split
        return text.split()


def tokenize_sentences(text: str) -> list:
    """
    Segment text into sentences.
    
    Args:
        text: Input text string
    
    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []
    try:
        return sent_tokenize(text)
    except Exception:
        # Fallback: split on period/question/exclamation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def get_word_tokens_no_punct(text: str) -> list:
    """
    Tokenize and remove punctuation tokens.
    
    Returns:
        List of word tokens (no punctuation)
    """
    tokens = tokenize_words(text)
    return [t for t in tokens if t.isalpha()]


def get_token_stats(text: str) -> dict:
    """
    Get comprehensive tokenization statistics.
    
    Returns:
        Dictionary with token counts and details
    """
    if not text:
        return {
            'word_tokens': [],
            'word_count': 0,
            'sentences': [],
            'sentence_count': 0,
            'avg_sentence_length': 0.0,
            'words_no_punct': [],
            'word_count_no_punct': 0
        }
    
    word_tokens = tokenize_words(text)
    sentences = tokenize_sentences(text)
    words_no_punct = get_word_tokens_no_punct(text)
    
    avg_sent_len = 0.0
    if sentences:
        words_per_sentence = [len(get_word_tokens_no_punct(s)) for s in sentences]
        avg_sent_len = sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0.0
    
    return {
        'word_tokens': word_tokens,
        'word_count': len(word_tokens),
        'sentences': sentences,
        'sentence_count': len(sentences),
        'avg_sentence_length': round(avg_sent_len, 1),
        'words_no_punct': words_no_punct,
        'word_count_no_punct': len(words_no_punct)
    }
