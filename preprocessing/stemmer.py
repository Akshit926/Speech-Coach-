"""
Stemming Module for ToastMasterAI
Implements stemming using NLTK PorterStemmer and SnowballStemmer.
"""

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer

# Ensure required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def get_porter_stemmer():
    """Return a PorterStemmer instance."""
    return PorterStemmer()


def get_snowball_stemmer(language='english'):
    """Return a SnowballStemmer instance."""
    return SnowballStemmer(language)


def stem_word(word: str, stemmer=None) -> str:
    """
    Stem a single word.
    
    Args:
        word: Input word
        stemmer: Stemmer instance (defaults to PorterStemmer)
    
    Returns:
        Stemmed word
    """
    if not word:
        return ""
    if stemmer is None:
        stemmer = get_porter_stemmer()
    return stemmer.stem(word)


def stem_words(words: list, stemmer=None) -> list:
    """
    Stem a list of words.
    
    Args:
        words: List of word strings
        stemmer: Stemmer instance (defaults to PorterStemmer)
    
    Returns:
        List of stemmed words
    """
    if not words:
        return []
    if stemmer is None:
        stemmer = get_porter_stemmer()
    return [stemmer.stem(w) for w in words]


def stem_text(text: str, stemmer=None) -> dict:
    """
    Stem all words in a text and return original-to-stem mapping.
    
    Returns:
        Dictionary with 'stems' list and 'mapping' dict
    """
    from preprocessing.tokenizer import get_word_tokens_no_punct
    
    if not text:
        return {'stems': [], 'mapping': {}}
    
    if stemmer is None:
        stemmer = get_porter_stemmer()
    
    words = get_word_tokens_no_punct(text.lower())
    stems = [stemmer.stem(w) for w in words]
    
    # Create word -> stem mapping
    mapping = {}
    for word, stem in zip(words, stems):
        if word not in mapping:
            mapping[word] = stem
    
    return {
        'stems': stems,
        'mapping': mapping
    }
