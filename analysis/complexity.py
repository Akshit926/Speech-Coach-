"""
Speech Complexity Score Module for ToastMasterAI
Custom NLP-based complexity scoring from 0-100.
"""

from collections import Counter
from preprocessing.tokenizer import tokenize_sentences, get_word_tokens_no_punct
from utils.helpers import get_complexity_label

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

# Common/simple words (high frequency in everyday English)
COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
    'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
    'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
    'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'very', 'thing', 'really', 'said', 'much'
}


def compute_complexity(text: str) -> dict:
    """
    Compute a comprehensive speech complexity score.
    
    Components:
    1. Average sentence length (longer = more complex)
    2. Vocabulary richness (TTR)
    3. Rare word usage
    4. Sentence length variation
    5. Word length distribution
    6. POS diversity
    
    Returns:
        Dictionary with overall score, component scores, label, and interpretation
    """
    if not text or len(text.strip()) < 20:
        return _empty_complexity()
    
    sentences = tokenize_sentences(text)
    words = get_word_tokens_no_punct(text.lower())
    
    if not words or not sentences:
        return _empty_complexity()
    
    # ─── Component 1: Average Sentence Length ─────────────────
    sent_lengths = [len(get_word_tokens_no_punct(s.lower())) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    
    # Normalize: 5-10 words/sentence = simple (20), 15-25 = moderate (50), 25+ = complex (80)
    if avg_sent_len <= 10:
        sent_score = 20
    elif avg_sent_len <= 20:
        sent_score = 20 + (avg_sent_len - 10) * 4  # 20-60
    elif avg_sent_len <= 30:
        sent_score = 60 + (avg_sent_len - 20) * 3  # 60-90
    else:
        sent_score = min(100, 90 + (avg_sent_len - 30))
    
    # ─── Component 2: Vocabulary Richness (TTR) ──────────────
    unique_words = set(words)
    ttr = len(unique_words) / len(words) * 100 if words else 0
    vocab_score = min(100, ttr * 1.5)
    
    # ─── Component 3: Rare Word Usage ─────────────────────────
    content_words = [w for w in words if w not in STOP_WORDS]
    rare_words = [w for w in content_words if w not in COMMON_WORDS and len(w) >= 6]
    rare_ratio = len(rare_words) / len(content_words) * 100 if content_words else 0
    rare_score = min(100, rare_ratio * 3)
    
    # ─── Component 4: Sentence Length Variation ───────────────
    if len(sent_lengths) > 1:
        mean = sum(sent_lengths) / len(sent_lengths)
        std = (sum((l - mean) ** 2 for l in sent_lengths) / len(sent_lengths)) ** 0.5
        variation_score = min(100, std * 5)
    else:
        variation_score = 0
    
    # ─── Component 5: Average Word Length ─────────────────────
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    # Normalize: 3-4 chars = simple, 5-6 = moderate, 7+ = complex
    word_len_score = min(100, max(0, (avg_word_len - 3) * 25))
    
    # ─── Combined Score ───────────────────────────────────────
    weights = {
        'sentence_length': 0.25,
        'vocabulary_richness': 0.25,
        'rare_words': 0.20,
        'sentence_variation': 0.15,
        'word_length': 0.15
    }
    
    component_scores = {
        'sentence_length': round(sent_score, 1),
        'vocabulary_richness': round(vocab_score, 1),
        'rare_words': round(rare_score, 1),
        'sentence_variation': round(variation_score, 1),
        'word_length': round(word_len_score, 1)
    }
    
    overall = sum(
        component_scores[k] * weights[k] for k in weights
    )
    overall = round(min(100, max(0, overall)), 1)
    
    label = get_complexity_label(overall)
    interpretation = _interpret_complexity(overall, component_scores, avg_sent_len, ttr, rare_ratio)
    
    return {
        'overall_score': overall,
        'label': label,
        'component_scores': component_scores,
        'weights': weights,
        'avg_sentence_length': round(avg_sent_len, 1),
        'vocabulary_richness_pct': round(ttr, 1),
        'rare_word_pct': round(rare_ratio, 1),
        'avg_word_length': round(avg_word_len, 1),
        'interpretation': interpretation,
        'total_words': len(words),
        'total_sentences': len(sentences),
        'rare_words_sample': list(set(rare_words))[:10]
    }


def _interpret_complexity(overall, components, avg_sent_len, ttr, rare_ratio):
    """Generate user-friendly complexity interpretation."""
    interps = []
    
    label = get_complexity_label(overall)
    interps.append(f"Speech Complexity: {label} ({overall}/100)")
    
    if overall < 30:
        interps.append("Your speech uses simple, accessible language — ideal for broad audiences and introductory speeches.")
    elif overall < 50:
        interps.append("Your speech has straightforward language with some complexity — good for most Toastmasters settings.")
    elif overall < 70:
        interps.append("Your speech has moderate complexity — suitable for informed audiences.")
    elif overall < 85:
        interps.append("Your speech uses advanced language. Ensure your audience can follow the complexity.")
    else:
        interps.append("Your speech is highly complex. Consider simplifying for better audience comprehension.")
    
    # Specific component feedback
    if avg_sent_len > 25:
        interps.append(f"Average sentence length is {avg_sent_len:.0f} words — consider breaking up longer sentences.")
    
    if rare_ratio > 20:
        interps.append("You use many advanced/rare words, which can be impressive but may reduce clarity.")
    elif rare_ratio < 5:
        interps.append("Consider incorporating a few more sophisticated vocabulary choices to elevate your speech.")
    
    return interps


def _empty_complexity():
    return {
        'overall_score': 0, 'label': 'Unknown',
        'component_scores': {}, 'weights': {},
        'avg_sentence_length': 0, 'vocabulary_richness_pct': 0,
        'rare_word_pct': 0, 'avg_word_length': 0,
        'interpretation': ['Not enough text for complexity analysis.'],
        'total_words': 0, 'total_sentences': 0, 'rare_words_sample': []
    }
