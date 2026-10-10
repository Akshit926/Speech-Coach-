"""
Lemmatization Module for ToastMasterAI
Implements lemmatization using NLTK WordNet and spaCy.
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure required NLTK data
for resource in ['wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}' if 'corpora' not in resource else resource)
    except LookupError:
        nltk.download(resource, quiet=True)


def get_wordnet_pos(treebank_tag: str):
    """
    Map POS treebank tag to WordNet POS tag for better lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default


def lemmatize_word(word: str, pos: str = 'n') -> str:
    """
    Lemmatize a single word.
    
    Args:
        word: Input word
        pos: WordNet POS tag (n, v, a, r)
    
    Returns:
        Lemmatized word
    """
    if not word:
        return ""
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower(), pos=pos)


def lemmatize_words(words: list) -> list:
    """
    Lemmatize a list of words using POS-aware lemmatization.
    
    Args:
        words: List of word strings
    
    Returns:
        List of lemmatized words
    """
    if not words:
        return []
    
    lemmatizer = WordNetLemmatizer()
    
    # POS tag the words for better lemmatization
    try:
        tagged = nltk.pos_tag(words)
        return [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag)) for word, tag in tagged]
    except Exception:
        return [lemmatizer.lemmatize(w.lower()) for w in words]


def lemmatize_text(text: str) -> dict:
    """
    Lemmatize all words in a text and return original-to-lemma mapping.
    
    Returns:
        Dictionary with 'lemmas' list and 'mapping' dict
    """
    from preprocessing.tokenizer import get_word_tokens_no_punct
    
    if not text:
        return {'lemmas': [], 'mapping': {}}
    
    words = get_word_tokens_no_punct(text.lower())
    lemmas = lemmatize_words(words)
    
    # Create word -> lemma mapping
    mapping = {}
    for word, lemma in zip(words, lemmas):
        if word not in mapping:
            mapping[word] = lemma
    
    return {
        'lemmas': lemmas,
        'mapping': mapping
    }


def get_word_alternatives(word: str) -> list:
    """
    Get synonyms from WordNet for vocabulary suggestions.
    
    Args:
        word: Input word
    
    Returns:
        List of synonym strings
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)[:10]
