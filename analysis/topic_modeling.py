"""
Topic Modeling Module for ToastMasterAI
Discovers hidden topics in speeches using LDA (Latent Dirichlet Allocation).
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = list(stopwords.words('english'))


def discover_topics(text: str, n_topics: int = 3, n_words: int = 6) -> dict:
    """
    Discover hidden topics in a speech using LDA.
    
    For a single speech, sentences are treated as documents.
    
    Args:
        text: Input speech text
        n_topics: Number of topics to extract
        n_words: Number of words per topic
    
    Returns:
        Dictionary with topics, distribution, and user-friendly labels
    """
    if not text or len(text.strip()) < 50:
        return _empty_topics()
    
    from preprocessing.tokenizer import tokenize_sentences
    sentences = tokenize_sentences(text)
    
    if len(sentences) < 3:
        return _empty_topics()
    
    # Adjust n_topics if too few sentences
    n_topics = min(n_topics, max(2, len(sentences) // 3))
    
    try:
        # Vectorize sentences
        vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            max_features=300,
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        
        doc_term_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        if doc_term_matrix.shape[1] < n_topics:
            n_topics = max(2, doc_term_matrix.shape[1])
        
        # Train LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        
        lda.fit(doc_term_matrix)
        
        # Extract topics
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_word_indices = topic.argsort()[::-1][:n_words]
            top_words = [feature_names[i] for i in top_word_indices]
            top_weights = [round(float(topic[i]), 3) for i in top_word_indices]
            
            topics.append({
                'id': idx + 1,
                'words': top_words,
                'weights': top_weights,
                'label': _generate_topic_label(top_words)
            })
        
        # Get document-topic distribution
        doc_topic_dist = lda.transform(doc_term_matrix)
        
        # Overall topic distribution
        avg_dist = doc_topic_dist.mean(axis=0)
        primary_topic_idx = np.argmax(avg_dist)
        
        topic_distribution = {}
        for i, weight in enumerate(avg_dist):
            topic_distribution[f"Topic {i+1}: {topics[i]['label']}"] = round(float(weight) * 100, 1)
        
        return {
            'topics': topics,
            'n_topics': n_topics,
            'primary_topic': topics[primary_topic_idx]['label'],
            'secondary_topic': topics[np.argsort(avg_dist)[-2]]['label'] if n_topics > 1 else '',
            'topic_distribution': topic_distribution,
            'interpretation': _interpret_topics(topics, avg_dist, primary_topic_idx)
        }
    
    except Exception as e:
        return {**_empty_topics(), 'error': str(e)}


def discover_topics_corpus(texts: list, n_topics: int = 5, n_words: int = 6) -> dict:
    """
    Discover topics across a corpus of speeches.
    
    Args:
        texts: List of speech texts
        n_topics: Number of topics
        n_words: Words per topic
    
    Returns:
        Dictionary with topics and per-document distributions
    """
    if not texts or len(texts) < 2:
        return _empty_topics()
    
    try:
        vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            max_features=500,
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        n_topics = min(n_topics, doc_term_matrix.shape[1])
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=30,
            learning_method='online'
        )
        
        lda.fit(doc_term_matrix)
        
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[::-1][:n_words]
            topics.append({
                'id': idx + 1,
                'words': [feature_names[i] for i in top_idx],
                'label': _generate_topic_label([feature_names[i] for i in top_idx])
            })
        
        # Per-document distribution
        doc_topic_dist = lda.transform(doc_term_matrix)
        per_document = []
        for i, dist in enumerate(doc_topic_dist):
            primary = np.argmax(dist)
            per_document.append({
                'primary_topic': topics[primary]['label'],
                'distribution': {topics[j]['label']: round(float(d) * 100, 1) for j, d in enumerate(dist)}
            })
        
        return {
            'topics': topics,
            'n_topics': n_topics,
            'per_document': per_document
        }
    except Exception:
        return _empty_topics()


def _generate_topic_label(words: list) -> str:
    """Generate a human-readable label for a topic based on its top words."""
    # Map common word clusters to topic labels
    topic_labels = {
        frozenset(['confidence', 'fear', 'speaking', 'stage', 'speech']): 'Confidence & Public Speaking',
        frozenset(['leader', 'leadership', 'team', 'lead', 'manage']): 'Leadership',
        frozenset(['learn', 'learning', 'growth', 'grow', 'experience']): 'Personal Growth',
        frozenset(['listen', 'communication', 'talk', 'speak', 'conversation']): 'Communication',
        frozenset(['fail', 'failure', 'success', 'try', 'mistake']): 'Resilience & Learning',
        frozenset(['story', 'life', 'journey', 'moment', 'day']): 'Personal Story',
        frozenset(['humor', 'laugh', 'funny', 'joke', 'comedy']): 'Humor & Entertainment',
        frozenset(['team', 'work', 'together', 'collaboration', 'group']): 'Teamwork',
    }
    
    word_set = set(w.lower() for w in words[:4])
    
    best_match = None
    best_overlap = 0
    for label_words, label in topic_labels.items():
        overlap = len(word_set & label_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = label
    
    if best_match and best_overlap >= 1:
        return best_match
    
    # Fallback: use top 2-3 words capitalized
    return ' & '.join(w.capitalize() for w in words[:3])


def _interpret_topics(topics: list, distribution: np.ndarray, primary_idx: int) -> list:
    """Generate user-friendly interpretation of discovered topics."""
    interps = []
    
    primary = topics[primary_idx]
    primary_pct = round(float(distribution[primary_idx]) * 100, 1)
    
    interps.append(f"Primary Topic: {primary['label']} ({primary_pct}% of speech content)")
    
    if len(topics) > 1:
        secondary_idx = np.argsort(distribution)[-2]
        secondary = topics[secondary_idx]
        secondary_pct = round(float(distribution[secondary_idx]) * 100, 1)
        interps.append(f"Secondary Topic: {secondary['label']} ({secondary_pct}%)")
    
    if primary_pct > 60:
        interps.append("Your speech is strongly focused on a single theme, which is good for message clarity.")
    elif primary_pct < 35:
        interps.append("Your speech covers multiple topics. Consider focusing more on your central theme.")
    
    return interps


def _empty_topics():
    return {
        'topics': [], 'n_topics': 0,
        'primary_topic': 'Unknown', 'secondary_topic': '',
        'topic_distribution': {},
        'interpretation': ['Not enough text for topic discovery.']
    }
