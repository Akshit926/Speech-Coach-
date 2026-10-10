"""
TF-IDF Analysis Module for ToastMasterAI
Extracts key vocabulary using TF-IDF from scikit-learn.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = list(stopwords.words('english'))


def extract_keywords_tfidf(text: str, top_n: int = 15) -> dict:
    """
    Extract important keywords from a speech using TF-IDF.
    
    For a single document, we split it into sentences and treat each as a 
    'document' in the corpus. This gives meaningful IDF values.
    
    Args:
        text: Input speech text
        top_n: Number of top keywords to return
    
    Returns:
        Dictionary with keywords, scores, and user-friendly explanation
    """
    if not text or len(text.strip()) < 20:
        return _empty_tfidf()
    
    from preprocessing.tokenizer import tokenize_sentences
    sentences = tokenize_sentences(text)
    
    if len(sentences) < 2:
        sentences = [text]
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            max_features=500,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # words with 3+ chars
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF score across all sentences
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Sort by score
        top_indices = avg_scores.argsort()[::-1][:top_n]
        
        keywords = []
        for idx in top_indices:
            if avg_scores[idx] > 0:
                keywords.append({
                    'word': feature_names[idx],
                    'score': round(float(avg_scores[idx]), 4),
                    'importance': _score_to_importance(float(avg_scores[idx]), float(avg_scores[top_indices[0]]))
                })
        
        # Generate explanation
        if keywords:
            top_words = [kw['word'] for kw in keywords[:5]]
            explanation = f"Your speech strongly focuses on: {', '.join(top_words)}."
        else:
            explanation = "Could not determine key vocabulary from this speech."
        
        return {
            'keywords': keywords,
            'top_words': [kw['word'] for kw in keywords],
            'explanation': explanation,
            'tfidf_matrix_shape': tfidf_matrix.shape,
            'vocabulary_size': len(feature_names),
            'feature_names': list(feature_names),
            'raw_scores': {feature_names[i]: round(float(avg_scores[i]), 4) 
                           for i in top_indices[:30] if avg_scores[i] > 0}
        }
    except Exception as e:
        return {**_empty_tfidf(), 'error': str(e)}


def compute_tfidf_for_corpus(texts: list, top_n: int = 10) -> dict:
    """
    Compute TF-IDF across a corpus of speeches.
    
    Args:
        texts: List of speech texts
        top_n: Number of top keywords per document
    
    Returns:
        Dictionary with per-document keywords and global vocabulary
    """
    if not texts:
        return {'per_document': [], 'vocabulary_size': 0}
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            max_features=1000,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        per_document = []
        for i in range(len(texts)):
            row = np.asarray(tfidf_matrix[i].todense()).flatten()
            top_idx = row.argsort()[::-1][:top_n]
            doc_keywords = [(feature_names[j], round(float(row[j]), 4)) for j in top_idx if row[j] > 0]
            per_document.append(doc_keywords)
        
        return {
            'per_document': per_document,
            'vocabulary_size': len(feature_names),
            'matrix_shape': tfidf_matrix.shape
        }
    except Exception:
        return {'per_document': [], 'vocabulary_size': 0}


def _score_to_importance(score: float, max_score: float) -> str:
    """Convert TF-IDF score to human-readable importance."""
    if max_score == 0:
        return "Low"
    ratio = score / max_score
    if ratio > 0.7:
        return "Very High"
    elif ratio > 0.5:
        return "High"
    elif ratio > 0.3:
        return "Medium"
    else:
        return "Low"


def _empty_tfidf():
    return {
        'keywords': [], 'top_words': [],
        'explanation': 'No text available for TF-IDF analysis.',
        'tfidf_matrix_shape': (0, 0), 'vocabulary_size': 0,
        'feature_names': [], 'raw_scores': {}
    }
