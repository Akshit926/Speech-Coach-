"""
Repetition Analyzer Module for ToastMasterAI
Detects repeated words, phrases, and sentence patterns.
"""

from collections import Counter
from preprocessing.tokenizer import tokenize_sentences, get_word_tokens_no_punct
from analysis.ngrams import analyze_ngrams

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def analyze_repetition(text: str) -> dict:
    """
    Comprehensive repetition analysis of the speech.
    
    Returns:
        Dictionary with repeated words, phrases, patterns, and suggestions
    """
    if not text:
        return _empty_repetition()
    
    words = get_word_tokens_no_punct(text.lower())
    sentences = tokenize_sentences(text)
    
    # ─── Repeated Words ───────────────────────────────────────
    # Content words only (no stop words)
    content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    word_freq = Counter(content_words)
    
    # Words repeated more than expected
    avg_freq = sum(word_freq.values()) / len(word_freq) if word_freq else 1
    repeated_words = {
        word: count for word, count in word_freq.most_common(30)
        if count >= max(3, avg_freq * 1.5)
    }
    
    # ─── Repeated Phrases (from n-grams) ──────────────────────
    ngram_data = analyze_ngrams(text)
    repeated_phrases = ngram_data.get('repeated_phrases', [])
    
    # ─── Sentence Starters ────────────────────────────────────
    starters = Counter()
    for sent in sentences:
        words_in_sent = get_word_tokens_no_punct(sent.lower())
        if len(words_in_sent) >= 2:
            starter = ' '.join(words_in_sent[:2])
            starters[starter] += 1
    
    repeated_starters = {
        starter: count for starter, count in starters.most_common(10)
        if count >= 3
    }
    
    # ─── Repetition Score ─────────────────────────────────────
    # Higher score = more repetition (not always bad)
    rep_score = _compute_repetition_score(
        repeated_words, repeated_phrases, repeated_starters,
        len(content_words), len(sentences)
    )
    
    # ─── Generate Suggestions ─────────────────────────────────
    suggestions = _generate_repetition_suggestions(
        repeated_words, repeated_phrases, repeated_starters, rep_score
    )
    
    return {
        'repeated_words': repeated_words,
        'repeated_phrases': repeated_phrases[:10],
        'repeated_starters': repeated_starters,
        'repetition_score': rep_score,
        'total_content_words': len(content_words),
        'unique_content_words': len(word_freq),
        'suggestions': suggestions,
        'word_frequencies': word_freq.most_common(20)
    }


def _compute_repetition_score(words, phrases, starters, total_words, total_sents):
    """
    Compute a repetition level from 0-100.
    0 = no repetition, 100 = extreme repetition.
    """
    if total_words == 0:
        return 0
    
    score = 0
    
    # Word repetition impact
    total_repeated_words = sum(words.values())
    word_rep_ratio = total_repeated_words / total_words if total_words > 0 else 0
    score += min(40, int(word_rep_ratio * 100))
    
    # Phrase repetition impact
    total_repeated_phrases = sum(p['count'] for p in phrases) if phrases else 0
    score += min(30, total_repeated_phrases * 3)
    
    # Starter repetition impact
    total_rep_starters = sum(starters.values())
    starter_ratio = total_rep_starters / total_sents if total_sents > 0 else 0
    score += min(30, int(starter_ratio * 60))
    
    return min(100, score)


def _generate_repetition_suggestions(words, phrases, starters, score):
    """Generate actionable suggestions based on repetition analysis."""
    suggestions = []
    
    # Most repeated word
    if words:
        top_word = list(words.items())[0]
        suggestions.append(
            f'The word "{top_word[0]}" appears {top_word[1]} times. '
            f'Consider using synonyms to add variety.'
        )
    
    # Repeated phrases
    if phrases:
        top_phrase = phrases[0]
        suggestions.append(
            f'The phrase "{top_phrase["phrase"]}" appears {top_phrase["count"]} times. '
            f'Consider varying the phrasing to maintain audience interest.'
        )
    
    # Repeated starters
    if starters:
        top_starter = list(starters.items())[0]
        suggestions.append(
            f'You start {top_starter[1]} sentences with "{top_starter[0]}". '
            f'Varying your sentence openings creates a more dynamic speaking flow.'
        )
    
    # Overall assessment
    if score < 20:
        suggestions.append("Your speech has minimal repetition — good variety in your language choices.")
    elif score < 40:
        suggestions.append("Your repetition level is moderate. Some repetition reinforces key points, which is good.")
    elif score < 60:
        suggestions.append("Your speech has notable repetition. Review which repetitions are intentional and which could be varied.")
    else:
        suggestions.append("High repetition detected. Reducing unnecessary repetition will make your speech more polished.")
    
    return suggestions


def _empty_repetition():
    return {
        'repeated_words': {}, 'repeated_phrases': [],
        'repeated_starters': {}, 'repetition_score': 0,
        'total_content_words': 0, 'unique_content_words': 0,
        'suggestions': ['No text for repetition analysis.'],
        'word_frequencies': []
    }
