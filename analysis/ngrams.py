"""
N-Gram Analysis Module for ToastMasterAI
Identifies frequently used phrases and repeated expressions.
"""

from collections import Counter
from preprocessing.tokenizer import get_word_tokens_no_punct

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def generate_ngrams(tokens: list, n: int) -> list:
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of word tokens
        n: Size of n-gram (1, 2, 3, etc.)
    
    Returns:
        List of n-gram tuples
    """
    if not tokens or n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def analyze_ngrams(text: str, max_n: int = 3) -> dict:
    """
    Comprehensive n-gram analysis for phrase and repetition detection.
    
    Args:
        text: Input text
        max_n: Maximum n-gram size (default: trigrams)
    
    Returns:
        Dictionary with unigrams, bigrams, trigrams, and interpretations
    """
    if not text:
        return _empty_ngrams()
    
    tokens = get_word_tokens_no_punct(text.lower())
    
    # Filter out stop words for content n-grams
    content_tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    results = {}
    
    # Unigrams (from content tokens)
    unigrams = Counter(content_tokens)
    results['unigrams'] = unigrams.most_common(20)
    
    # Bigrams (from all tokens to capture natural phrases)
    bigrams = Counter(generate_ngrams(tokens, 2))
    # Filter bigrams that are just stop words
    meaningful_bigrams = {
        bg: count for bg, count in bigrams.items()
        if not all(w in STOP_WORDS for w in bg) and count > 1
    }
    results['bigrams'] = Counter(meaningful_bigrams).most_common(20)
    
    # Trigrams
    trigrams = Counter(generate_ngrams(tokens, 3))
    meaningful_trigrams = {
        tg: count for tg, count in trigrams.items()
        if not all(w in STOP_WORDS for w in tg) and count > 1
    }
    results['trigrams'] = Counter(meaningful_trigrams).most_common(15)
    
    # Format for display
    repeated_phrases = []
    for gram, count in results['bigrams'][:10]:
        phrase = ' '.join(gram)
        repeated_phrases.append({'phrase': phrase, 'count': count, 'type': 'bigram'})
    
    for gram, count in results['trigrams'][:10]:
        phrase = ' '.join(gram)
        repeated_phrases.append({'phrase': phrase, 'count': count, 'type': 'trigram'})
    
    # Sort by count descending
    repeated_phrases.sort(key=lambda x: x['count'], reverse=True)
    
    # Generate interpretation
    interpretation = _interpret_ngrams(repeated_phrases)
    
    return {
        'unigrams': results['unigrams'],
        'bigrams': results['bigrams'],
        'trigrams': results['trigrams'],
        'repeated_phrases': repeated_phrases[:15],
        'interpretation': interpretation,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens))
    }


def _interpret_ngrams(phrases: list) -> list:
    """Generate user-friendly interpretation of n-gram results."""
    interpretations = []
    
    if not phrases:
        interpretations.append("No significantly repeated phrases detected in your speech.")
        return interpretations
    
    high_repeat = [p for p in phrases if p['count'] >= 4]
    moderate_repeat = [p for p in phrases if 2 <= p['count'] < 4]
    
    if high_repeat:
        top_phrase = high_repeat[0]
        interpretations.append(
            f'The phrase "{top_phrase["phrase"]}" appears {top_phrase["count"]} times. '
            f'This high repetition may be intentional for emphasis, or it could indicate overuse.'
        )
    
    if moderate_repeat:
        phrases_text = ', '.join([f'"{p["phrase"]}"' for p in moderate_repeat[:3]])
        interpretations.append(
            f'Moderately repeated phrases include {phrases_text}. '
            f'Consider whether each repetition adds value or could be varied.'
        )
    
    if len(phrases) > 10:
        interpretations.append(
            "Your speech contains several recurring patterns. "
            "While some repetition can reinforce key ideas, too much can reduce engagement."
        )
    
    return interpretations


def _empty_ngrams():
    return {
        'unigrams': [], 'bigrams': [], 'trigrams': [],
        'repeated_phrases': [],
        'interpretation': ['No text available for n-gram analysis.'],
        'total_tokens': 0, 'unique_tokens': 0
    }
