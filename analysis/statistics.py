"""
Speech Statistics Module for ToastMasterAI
Word count, sentence count, vocabulary richness, speaking time.
"""

from collections import Counter
from preprocessing.tokenizer import tokenize_words, tokenize_sentences, get_word_tokens_no_punct
from utils.helpers import estimate_speaking_time, safe_divide, FILLER_WORDS, OVERUSED_WORDS

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def compute_statistics(text: str, wpm: int = 140) -> dict:
    """
    Compute comprehensive speech statistics.
    
    Args:
        text: The speech text (original, not cleaned)
        wpm: Words per minute for speaking time estimate
    
    Returns:
        Dictionary of speech statistics
    """
    if not text or not isinstance(text, str):
        return _empty_stats()
    
    sentences = tokenize_sentences(text)
    words = get_word_tokens_no_punct(text.lower())
    all_tokens = tokenize_words(text)
    
    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = set(words)
    unique_count = len(unique_words)
    
    # Content words (no stop words)
    content_words = [w for w in words if w not in STOP_WORDS]
    
    # Vocabulary richness (Type-Token Ratio)
    ttr = safe_divide(unique_count, word_count) * 100
    
    # Average sentence length
    avg_sent_len = safe_divide(word_count, sentence_count)
    
    # Most frequent words (content words only)
    word_freq = Counter(content_words)
    most_common = word_freq.most_common(20)
    
    # Detect filler words
    filler_counts = {}
    text_lower = text.lower()
    for filler in FILLER_WORDS:
        count = text_lower.split().count(filler) if ' ' not in filler else text_lower.count(filler)
        if count > 0:
            filler_counts[filler] = count
    
    # Detect overused words
    overused_counts = {}
    for word in OVERUSED_WORDS:
        count = words.count(word)
        if count >= 2:
            overused_counts[word] = count
    
    # Speaking time
    speaking_time = estimate_speaking_time(word_count, wpm)
    
    # Sentence length variation
    sent_lengths = [len(get_word_tokens_no_punct(s.lower())) for s in sentences]
    sent_length_std = 0.0
    if len(sent_lengths) > 1:
        mean_sl = sum(sent_lengths) / len(sent_lengths)
        sent_length_std = (sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)) ** 0.5
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'unique_word_count': unique_count,
        'vocabulary_richness': round(ttr, 1),
        'avg_sentence_length': round(avg_sent_len, 1),
        'speaking_time': speaking_time,
        'wpm': wpm,
        'most_common_words': most_common,
        'content_words': content_words,
        'filler_words': filler_counts,
        'overused_words': overused_counts,
        'sentence_lengths': sent_lengths,
        'sentence_length_std': round(sent_length_std, 1),
        'sentences': sentences,
        'words': words,
        'all_tokens': all_tokens
    }


def _empty_stats():
    """Return empty statistics dictionary."""
    return {
        'word_count': 0, 'sentence_count': 0, 'unique_word_count': 0,
        'vocabulary_richness': 0.0, 'avg_sentence_length': 0.0,
        'speaking_time': '0 min 0 sec', 'wpm': 140,
        'most_common_words': [], 'content_words': [],
        'filler_words': {}, 'overused_words': {},
        'sentence_lengths': [], 'sentence_length_std': 0.0,
        'sentences': [], 'words': [], 'all_tokens': []
    }


def get_vocabulary_coach_data(stats: dict) -> dict:
    """
    Generate Vocabulary Coach insights from statistics.
    
    Returns:
        Dictionary with vocabulary coaching data
    """
    suggestions = []
    
    # Filler word suggestions
    for word, count in stats.get('filler_words', {}).items():
        if count >= 3:
            suggestions.append(
                f'You used "{word}" {count} times. Consider reducing its usage for a more polished delivery.'
            )
    
    # Overused word suggestions
    for word, count in stats.get('overused_words', {}).items():
        suggestions.append(
            f'The word "{word}" appears {count} times. Consider replacing some occurrences with more specific alternatives.'
        )
    
    # Vocabulary richness feedback
    ttr = stats.get('vocabulary_richness', 0)
    if ttr < 40:
        suggestions.append(
            'Your vocabulary diversity is low. Try using more varied words to make your speech more engaging.'
        )
    elif ttr > 75:
        suggestions.append(
            'Excellent vocabulary diversity! Your word choices are varied and expressive.'
        )
    
    return {
        'top_words': stats.get('most_common_words', [])[:15],
        'filler_words': stats.get('filler_words', {}),
        'overused_words': stats.get('overused_words', {}),
        'vocabulary_richness': stats.get('vocabulary_richness', 0),
        'unique_words': stats.get('unique_word_count', 0),
        'total_words': stats.get('word_count', 0),
        'suggestions': suggestions
    }
