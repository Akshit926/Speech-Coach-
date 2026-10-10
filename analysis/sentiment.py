"""
Sentiment Analysis Module for ToastMasterAI
Analyzes sentiment at whole-speech and sentence levels.
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from preprocessing.tokenizer import tokenize_sentences


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of the entire speech and per sentence.
    
    Returns:
        Dictionary with overall sentiment, sentence sentiments, and timeline
    """
    if not text:
        return _empty_sentiment()
    
    sia = SentimentIntensityAnalyzer()
    
    # Overall speech sentiment
    overall_scores = sia.polarity_scores(text)
    overall_label = _classify_sentiment(overall_scores['compound'])
    
    # Sentence-level sentiment
    sentences = tokenize_sentences(text)
    sentence_sentiments = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for i, sent in enumerate(sentences):
        scores = sia.polarity_scores(sent)
        label = _classify_sentiment(scores['compound'])
        
        sentence_sentiments.append({
            'sentence': sent,
            'index': i,
            'compound': round(scores['compound'], 3),
            'positive': round(scores['pos'], 3),
            'negative': round(scores['neg'], 3),
            'neutral': round(scores['neu'], 3),
            'label': label
        })
        
        if label == 'Positive':
            positive_count += 1
        elif label == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
    
    total = len(sentences) if sentences else 1
    
    # Sentiment distribution
    distribution = {
        'Positive': round(positive_count / total * 100, 1),
        'Neutral': round(neutral_count / total * 100, 1),
        'Negative': round(negative_count / total * 100, 1)
    }
    
    # Sentiment timeline (compound scores over the speech)
    timeline = [s['compound'] for s in sentence_sentiments]
    
    # Interpretation
    interpretation = _interpret_sentiment(overall_label, distribution, timeline)
    
    return {
        'overall_sentiment': overall_label,
        'overall_scores': {
            'compound': round(overall_scores['compound'], 3),
            'positive': round(overall_scores['pos'], 3),
            'negative': round(overall_scores['neg'], 3),
            'neutral': round(overall_scores['neu'], 3)
        },
        'distribution': distribution,
        'sentence_sentiments': sentence_sentiments,
        'timeline': timeline,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_sentences': total,
        'interpretation': interpretation,
        'most_positive': max(sentence_sentiments, key=lambda x: x['compound'])['sentence'] if sentence_sentiments else '',
        'most_negative': min(sentence_sentiments, key=lambda x: x['compound'])['sentence'] if sentence_sentiments else ''
    }


def _classify_sentiment(compound: float) -> str:
    """Classify sentiment based on compound score."""
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def _interpret_sentiment(overall: str, distribution: dict, timeline: list) -> list:
    """Generate user-friendly sentiment interpretation."""
    interps = []
    
    interps.append(f"Overall Sentiment: {overall}")
    
    pos = distribution.get('Positive', 0)
    neg = distribution.get('Negative', 0)
    
    if pos > 60:
        interps.append("Your speech is predominantly positive and uplifting — excellent for motivational speeches.")
    elif neg > 30:
        interps.append("Your speech contains significant negative sentiment. This could be effective for contrast or storytelling, or may need adjustment.")
    else:
        interps.append("Your speech has a balanced emotional tone.")
    
    # Analyze trajectory
    if len(timeline) >= 4:
        first_quarter = sum(timeline[:len(timeline)//4]) / (len(timeline)//4)
        last_quarter = sum(timeline[-(len(timeline)//4):]) / (len(timeline)//4)
        
        if last_quarter > first_quarter + 0.1:
            interps.append("Your speech builds toward a positive ending — a strong persuasive pattern.")
        elif first_quarter > last_quarter + 0.1:
            interps.append("Your speech becomes less positive toward the end. Consider ending on a more uplifting note.")
    
    return interps


def _empty_sentiment():
    return {
        'overall_sentiment': 'Unknown',
        'overall_scores': {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0},
        'distribution': {'Positive': 0, 'Neutral': 0, 'Negative': 0},
        'sentence_sentiments': [], 'timeline': [],
        'positive_count': 0, 'negative_count': 0, 'neutral_count': 0,
        'total_sentences': 0,
        'interpretation': ['No text for sentiment analysis.'],
        'most_positive': '', 'most_negative': ''
    }
