"""
N-Gram Language Model Module for ToastMasterAI
Implements MLE and Laplace-smoothed n-gram probability models.
"""

from collections import Counter, defaultdict
from preprocessing.tokenizer import get_word_tokens_no_punct
import math


class NgramLanguageModel:
    """
    N-gram language model with Maximum Likelihood Estimation and Laplace smoothing.
    """
    
    def __init__(self, n: int = 3, smoothing: str = 'laplace'):
        """
        Args:
            n: N-gram size (1=unigram, 2=bigram, 3=trigram)
            smoothing: 'none' for MLE, 'laplace' for add-1 smoothing
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()
        self.total_tokens = 0
        self.is_trained = False
    
    def train(self, text: str):
        """Train the model on a text corpus."""
        tokens = ['<s>'] * (self.n - 1) + get_word_tokens_no_punct(text.lower()) + ['</s>']
        self.vocabulary = set(tokens)
        self.total_tokens = len(tokens)
        
        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
        
        self.is_trained = True
    
    def probability(self, word: str, context: tuple) -> float:
        """
        Calculate P(word | context) using MLE or Laplace smoothing.
        
        Args:
            word: Target word
            context: Tuple of preceding words
        
        Returns:
            Probability value
        """
        if not self.is_trained:
            return 0.0
        
        ngram = context + (word,)
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context]
        V = len(self.vocabulary)
        
        if self.smoothing == 'laplace':
            return (count_ngram + 1) / (count_context + V)
        else:
            # MLE
            if count_context == 0:
                return 0.0
            return count_ngram / count_context
    
    def sentence_probability(self, sentence: str) -> float:
        """
        Calculate the probability of a sentence.
        
        Returns:
            Log probability of the sentence
        """
        tokens = ['<s>'] * (self.n - 1) + get_word_tokens_no_punct(sentence.lower()) + ['</s>']
        log_prob = 0.0
        
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            prob = self.probability(word, context)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += -float('inf')
        
        return log_prob
    
    def perplexity(self, sentence: str) -> float:
        """
        Calculate perplexity of a sentence (lower = more natural).
        
        Returns:
            Perplexity score
        """
        tokens = get_word_tokens_no_punct(sentence.lower())
        if not tokens:
            return float('inf')
        
        log_prob = self.sentence_probability(sentence)
        
        if log_prob == -float('inf'):
            return float('inf')
        
        N = len(tokens)
        try:
            return math.exp(-log_prob / N)
        except (OverflowError, ZeroDivisionError):
            return float('inf')


def compute_language_pattern_score(text: str) -> dict:
    """
    Compute a user-friendly 'Language Pattern Score' using n-gram models.
    
    Trains on the speech itself and evaluates individual sentences
    to find patterns that fit or diverge from the speech's language patterns.
    
    Returns:
        Dictionary with score, natural/unusual sentences, and interpretation
    """
    if not text or len(text.strip()) < 50:
        return {
            'score': 0,
            'interpretation': 'Text is too short for language pattern analysis.',
            'natural_sentences': [],
            'unusual_sentences': [],
            'model_info': {}
        }
    
    from preprocessing.tokenizer import tokenize_sentences
    
    # Train trigram model on the speech
    model = NgramLanguageModel(n=3, smoothing='laplace')
    model.train(text)
    
    sentences = tokenize_sentences(text)
    
    if len(sentences) < 3:
        return {
            'score': 50,
            'interpretation': 'Too few sentences for detailed pattern analysis.',
            'natural_sentences': [],
            'unusual_sentences': [],
            'model_info': {'n': 3, 'vocabulary_size': len(model.vocabulary)}
        }
    
    # Score each sentence
    sentence_scores = []
    for sent in sentences:
        perp = model.perplexity(sent)
        if perp != float('inf') and perp > 0:
            sentence_scores.append((sent, perp))
    
    if not sentence_scores:
        return {
            'score': 50,
            'interpretation': 'Could not compute language pattern scores.',
            'natural_sentences': [],
            'unusual_sentences': [],
            'model_info': {'n': 3, 'vocabulary_size': len(model.vocabulary)}
        }
    
    # Sort by perplexity (low = natural, high = unusual)
    sentence_scores.sort(key=lambda x: x[1])
    
    # Natural sentences (low perplexity — common patterns)
    natural = [(s, round(p, 2)) for s, p in sentence_scores[:3]]
    
    # Unusual sentences (high perplexity — unique/uncommon phrasing)
    unusual = [(s, round(p, 2)) for s, p in sentence_scores[-3:]]
    
    # Compute an overall score (0-100)
    # Lower average perplexity = more consistent language patterns
    avg_perp = sum(p for _, p in sentence_scores) / len(sentence_scores)
    # Normalize to 0-100 (heuristic scaling)
    score = max(0, min(100, int(100 - min(avg_perp, 100))))
    
    interpretation = _interpret_language_score(score)
    
    return {
        'score': score,
        'interpretation': interpretation,
        'natural_sentences': natural,
        'unusual_sentences': unusual,
        'avg_perplexity': round(avg_perp, 2),
        'model_info': {
            'n': 3,
            'vocabulary_size': len(model.vocabulary),
            'total_ngrams': sum(model.ngram_counts.values()),
            'smoothing': 'Laplace (Add-1)'
        }
    }


def _interpret_language_score(score: int) -> str:
    """Human-readable interpretation of language pattern score."""
    if score >= 80:
        return "Your speech uses very consistent and natural language patterns."
    elif score >= 60:
        return "Your speech has good language consistency with some varied phrasing."
    elif score >= 40:
        return "Your speech shows moderate language pattern variety."
    elif score >= 20:
        return "Your speech has quite diverse language patterns, which may indicate creative or complex phrasing."
    else:
        return "Your speech uses highly varied language patterns."
