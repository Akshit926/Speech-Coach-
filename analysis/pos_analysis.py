"""
POS Analysis Module for ToastMasterAI
Part-of-Speech tagging and speaking style interpretation.
"""

import nltk
from collections import Counter

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize

# POS tag categories for interpretation
POS_CATEGORIES = {
    'Nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
    'Verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'Adjectives': ['JJ', 'JJR', 'JJS'],
    'Adverbs': ['RB', 'RBR', 'RBS'],
    'Pronouns': ['PRP', 'PRP$', 'WP', 'WP$'],
    'Prepositions': ['IN'],
    'Conjunctions': ['CC'],
    'Determiners': ['DT', 'PDT', 'WDT'],
    'Modals': ['MD']
}


def analyze_pos(text: str) -> dict:
    """
    Perform POS tagging and generate speaking style analysis.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with POS distribution, interpretation, and raw tags
    """
    if not text:
        return _empty_pos()
    
    try:
        tokens = word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        return _empty_pos()
    
    # Count POS categories
    pos_counts = Counter()
    category_counts = {cat: 0 for cat in POS_CATEGORIES}
    
    for word, tag in tagged:
        if not word.isalpha():
            continue
        pos_counts[tag] += 1
        for category, tags in POS_CATEGORIES.items():
            if tag in tags:
                category_counts[category] += 1
                break
    
    total = sum(category_counts.values())
    
    # Calculate percentages
    category_percentages = {}
    for cat, count in category_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        category_percentages[cat] = round(pct, 1)
    
    # Generate interpretation
    interpretation = _interpret_pos_distribution(category_percentages)
    
    # Get sample tagged words for NLP Lab Mode
    sample_tags = [(w, t) for w, t in tagged if w.isalpha()][:50]
    
    return {
        'category_counts': category_counts,
        'category_percentages': category_percentages,
        'total_tagged': total,
        'interpretation': interpretation,
        'sample_tags': sample_tags,
        'raw_pos_counts': dict(pos_counts.most_common(20))
    }


def _interpret_pos_distribution(percentages: dict) -> list:
    """Generate human-readable interpretation of POS distribution."""
    interpretations = []
    
    verbs = percentages.get('Verbs', 0)
    nouns = percentages.get('Nouns', 0)
    adj = percentages.get('Adjectives', 0)
    adv = percentages.get('Adverbs', 0)
    pronouns = percentages.get('Pronouns', 0)
    
    # Verb analysis
    if verbs > 20:
        interpretations.append(
            "Your speech uses a high proportion of action-oriented verbs, which creates an active and dynamic speaking style."
        )
    elif verbs < 10:
        interpretations.append(
            "Your speech has relatively few verbs. Consider using more action words to make your message more dynamic."
        )
    
    # Noun analysis
    if nouns > 30:
        interpretations.append(
            "Your speech is noun-heavy, which suggests a focus on concrete ideas, people, and things."
        )
    
    # Adjective analysis
    if adj > 10:
        interpretations.append(
            "Good use of adjectives — your speech includes descriptive language that helps paint vivid pictures."
        )
    elif adj < 5:
        interpretations.append(
            "Consider adding more descriptive adjectives to make your speech more vivid and engaging."
        )
    
    # Pronoun analysis
    if pronouns > 15:
        interpretations.append(
            "High pronoun usage indicates a personal, conversational tone — great for connecting with the audience."
        )
    
    # Adverb analysis
    if adv > 8:
        interpretations.append(
            "Your speech includes many adverbs, adding nuance to your statements."
        )
    
    # Overall style
    if verbs > nouns:
        interpretations.append(
            "Overall, your speaking style is action-oriented — you focus on what people do rather than what things are."
        )
    elif nouns > verbs * 1.5:
        interpretations.append(
            "Your speaking style is descriptive and informational — you focus on ideas and concepts."
        )
    
    if not interpretations:
        interpretations.append("Your speech has a balanced distribution of parts of speech.")
    
    return interpretations


def _empty_pos():
    """Return empty POS analysis."""
    return {
        'category_counts': {cat: 0 for cat in POS_CATEGORIES},
        'category_percentages': {cat: 0.0 for cat in POS_CATEGORIES},
        'total_tagged': 0,
        'interpretation': ['No text available for POS analysis.'],
        'sample_tags': [],
        'raw_pos_counts': {}
    }
