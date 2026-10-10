"""
Word Embeddings & Semantic Similarity Module for ToastMasterAI
Uses spaCy word vectors for semantic analysis and speech comparison.
"""

import numpy as np
from analysis.ner import get_nlp
from preprocessing.tokenizer import tokenize_sentences


def get_text_vector(text: str):
    """
    Get the average word vector for a piece of text using spaCy.
    
    Returns:
        numpy array vector or None
    """
    nlp = get_nlp()
    if nlp is None or not text:
        return None
    
    try:
        doc = nlp(text)
        if doc.vector is not None and np.any(doc.vector):
            return doc.vector
        return None
    except Exception:
        return None


def cosine_similarity(vec1, vec2) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns:
        Similarity score between -1 and 1
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


def compute_speech_similarity(text1: str, text2: str) -> dict:
    """
    Compare two speeches using semantic similarity.
    
    Returns:
        Dictionary with similarity score and interpretation
    """
    vec1 = get_text_vector(text1)
    vec2 = get_text_vector(text2)
    
    if vec1 is None or vec2 is None:
        return {
            'similarity': 0.0,
            'percentage': 0.0,
            'interpretation': 'Could not compute semantic similarity. Ensure spaCy model is installed.'
        }
    
    sim = cosine_similarity(vec1, vec2)
    pct = round(max(0, sim) * 100, 1)
    
    if pct > 80:
        interp = "These speeches are very similar in content and theme."
    elif pct > 60:
        interp = "These speeches share significant thematic overlap."
    elif pct > 40:
        interp = "These speeches have some common themes but cover different ground."
    elif pct > 20:
        interp = "These speeches are quite different in their subject matter."
    else:
        interp = "These speeches cover very different topics."
    
    return {
        'similarity': round(sim, 4),
        'percentage': pct,
        'interpretation': interp
    }


def analyze_section_coherence(text: str) -> dict:
    """
    Analyze semantic coherence between different parts of the speech.
    Compares opening vs conclusion, and tracks semantic flow.
    
    Returns:
        Dictionary with section comparisons and coherence scores
    """
    sentences = tokenize_sentences(text)
    
    if len(sentences) < 4:
        return {
            'opening_conclusion_similarity': 0.0,
            'avg_coherence': 0.0,
            'interpretation': 'Speech is too short for section coherence analysis.',
            'section_vectors': []
        }
    
    # Split into roughly 4 parts
    n = len(sentences)
    quarter = max(1, n // 4)
    
    opening = ' '.join(sentences[:quarter])
    middle1 = ' '.join(sentences[quarter:2*quarter])
    middle2 = ' '.join(sentences[2*quarter:3*quarter])
    conclusion = ' '.join(sentences[3*quarter:])
    
    sections = {
        'Opening': opening,
        'Body (First Half)': middle1,
        'Body (Second Half)': middle2,
        'Conclusion': conclusion
    }
    
    vectors = {}
    for name, section_text in sections.items():
        vectors[name] = get_text_vector(section_text)
    
    # Opening vs Conclusion similarity
    open_concl_sim = cosine_similarity(vectors.get('Opening'), vectors.get('Conclusion'))
    
    # Adjacent section coherence
    section_names = list(sections.keys())
    coherence_scores = []
    for i in range(len(section_names) - 1):
        sim = cosine_similarity(vectors.get(section_names[i]), vectors.get(section_names[i+1]))
        coherence_scores.append(sim)
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    
    # Interpretation
    interpretations = []
    if open_concl_sim > 0.7:
        interpretations.append("Your opening and conclusion are well-connected thematically, creating a cohesive circular structure.")
    elif open_concl_sim > 0.4:
        interpretations.append("Your opening and conclusion share some thematic connection.")
    else:
        interpretations.append("Your conclusion could connect more strongly back to your opening theme.")
    
    if avg_coherence > 0.7:
        interpretations.append("Your speech flows smoothly from one section to the next.")
    elif avg_coherence < 0.4:
        interpretations.append("There are noticeable thematic shifts between sections. Consider smoother transitions.")
    
    return {
        'opening_conclusion_similarity': round(open_concl_sim * 100, 1),
        'avg_coherence': round(avg_coherence * 100, 1),
        'section_similarities': {
            f'{section_names[i]} → {section_names[i+1]}': round(coherence_scores[i] * 100, 1)
            for i in range(len(coherence_scores))
        },
        'interpretation': interpretations
    }


def find_similar_words(word: str, top_n: int = 10) -> list:
    """
    Find semantically similar words using spaCy vectors.
    
    Returns:
        List of similar words (limited by spaCy's small model vocabulary)
    """
    nlp = get_nlp()
    if nlp is None or not word:
        return []
    
    try:
        word_doc = nlp(word)
        if not word_doc.vector.any():
            return []
        
        # With small model, similarity is limited, so we return empty
        # In a production system, this would use a larger model
        return []
    except Exception:
        return []
