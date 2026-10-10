"""
Emotion Detection Module for ToastMasterAI
Detects emotions like joy, sadness, fear, anger, surprise, trust, anticipation.
Uses a rule-based approach with emotion lexicons.
"""

from collections import Counter, defaultdict
from preprocessing.tokenizer import tokenize_sentences, get_word_tokens_no_punct

# NRC-style emotion lexicon (curated subset for academic demonstration)
# In production, load the full NRC Emotion Lexicon
EMOTION_LEXICON = {
    # Joy words
    'happy': 'joy', 'joy': 'joy', 'love': 'joy', 'wonderful': 'joy',
    'beautiful': 'joy', 'celebrate': 'joy', 'delight': 'joy', 'smile': 'joy',
    'laugh': 'joy', 'amazing': 'joy', 'great': 'joy', 'excellent': 'joy',
    'fantastic': 'joy', 'blessed': 'joy', 'grateful': 'joy', 'proud': 'joy',
    'success': 'joy', 'accomplish': 'joy', 'win': 'joy', 'triumph': 'joy',
    'extraordinary': 'joy', 'brilliant': 'joy', 'magnificent': 'joy',
    'clap': 'joy', 'cheer': 'joy', 'exciting': 'joy', 'thrilled': 'joy',
    'remarkable': 'joy', 'superpower': 'joy', 'fun': 'joy',
    
    # Sadness words
    'sad': 'sadness', 'cry': 'sadness', 'pain': 'sadness', 'loss': 'sadness',
    'lonely': 'sadness', 'grief': 'sadness', 'miss': 'sadness', 'hurt': 'sadness',
    'suffer': 'sadness', 'tears': 'sadness', 'sorrow': 'sadness', 'regret': 'sadness',
    'devastated': 'sadness', 'miserable': 'sadness', 'depressed': 'sadness',
    'depression': 'sadness', 'loneliness': 'sadness', 'sting': 'sadness',
    'struggled': 'sadness', 'struggling': 'sadness', 'worst': 'sadness',
    
    # Fear words
    'fear': 'fear', 'afraid': 'fear', 'scared': 'fear', 'terrified': 'fear',
    'panic': 'fear', 'nervous': 'fear', 'anxious': 'fear', 'worry': 'fear',
    'dread': 'fear', 'terror': 'fear', 'horror': 'fear', 'frighten': 'fear',
    'threat': 'fear', 'danger': 'fear', 'paralyzing': 'fear', 'trembling': 'fear',
    'uncomfortable': 'fear', 'anxiety': 'fear', 'intimidating': 'fear',
    
    # Anger words
    'angry': 'anger', 'rage': 'anger', 'hate': 'anger', 'furious': 'anger',
    'frustrate': 'anger', 'frustrated': 'anger', 'annoy': 'anger', 'irritate': 'anger',
    'outrage': 'anger', 'hostile': 'anger', 'resent': 'anger', 'bitter': 'anger',
    'fights': 'anger', 'fighting': 'anger', 'argue': 'anger', 'blame': 'anger',
    'disaster': 'anger', 'nightmare': 'anger',
    
    # Surprise words
    'surprise': 'surprise', 'shocked': 'surprise', 'amazed': 'surprise',
    'astonished': 'surprise', 'unexpected': 'surprise', 'incredible': 'surprise',
    'unbelievable': 'surprise', 'stunning': 'surprise', 'remarkable': 'surprise',
    'wow': 'surprise', 'fascinating': 'surprise', 'curious': 'surprise',
    'discovered': 'surprise', 'realized': 'surprise', 'suddenly': 'surprise',
    
    # Trust words
    'trust': 'trust', 'believe': 'trust', 'faith': 'trust', 'honest': 'trust',
    'reliable': 'trust', 'loyal': 'trust', 'confident': 'trust', 'confidence': 'trust',
    'safe': 'trust', 'support': 'trust', 'respect': 'trust', 'genuine': 'trust',
    'authentic': 'trust', 'integrity': 'trust', 'mentor': 'trust', 'mentored': 'trust',
    'cherish': 'trust', 'connection': 'trust', 'empathy': 'trust',
    
    # Anticipation words
    'hope': 'anticipation', 'expect': 'anticipation', 'dream': 'anticipation',
    'plan': 'anticipation', 'future': 'anticipation', 'goal': 'anticipation',
    'imagine': 'anticipation', 'wish': 'anticipation', 'aspire': 'anticipation',
    'potential': 'anticipation', 'opportunity': 'anticipation', 'vision': 'anticipation',
    'challenge': 'anticipation', 'encourage': 'anticipation', 'inspire': 'anticipation',
    'transform': 'anticipation', 'purpose': 'anticipation', 'mission': 'anticipation',
}


def detect_emotions(text: str) -> dict:
    """
    Detect emotions in the speech using a lexicon-based approach.
    
    Returns:
        Dictionary with emotion distribution, journey, and interpretation
    """
    if not text:
        return _empty_emotions()
    
    sentences = tokenize_sentences(text)
    
    # Overall emotion counts
    emotion_counts = Counter()
    total_emotion_words = 0
    
    # Per-sentence emotions for the journey
    sentence_emotions = []
    
    for i, sent in enumerate(sentences):
        words = get_word_tokens_no_punct(sent.lower())
        sent_emotions = Counter()
        
        for word in words:
            if word in EMOTION_LEXICON:
                emotion = EMOTION_LEXICON[word]
                sent_emotions[emotion] += 1
                emotion_counts[emotion] += 1
                total_emotion_words += 1
        
        # Determine dominant emotion for this sentence
        if sent_emotions:
            dominant = sent_emotions.most_common(1)[0][0]
        else:
            dominant = 'neutral'
        
        sentence_emotions.append({
            'sentence': sent,
            'index': i,
            'dominant_emotion': dominant,
            'emotions': dict(sent_emotions)
        })
    
    # Overall dominant emotion
    if emotion_counts:
        dominant_overall = emotion_counts.most_common(1)[0][0]
    else:
        dominant_overall = 'neutral'
    
    # Emotion distribution as percentages
    distribution = {}
    for emotion in ['joy', 'sadness', 'fear', 'anger', 'surprise', 'trust', 'anticipation']:
        count = emotion_counts.get(emotion, 0)
        pct = (count / total_emotion_words * 100) if total_emotion_words > 0 else 0
        distribution[emotion.capitalize()] = round(pct, 1)
    
    # Build emotional journey (simplified into speech sections)
    journey = _build_emotional_journey(sentence_emotions)
    
    # Interpretation
    interpretation = _interpret_emotions(dominant_overall, distribution, journey)
    
    return {
        'dominant_emotion': dominant_overall.capitalize(),
        'emotion_counts': dict(emotion_counts),
        'distribution': distribution,
        'total_emotion_words': total_emotion_words,
        'sentence_emotions': sentence_emotions,
        'journey': journey,
        'interpretation': interpretation,
        'journey_labels': [j['emotion'].capitalize() for j in journey]
    }


def _build_emotional_journey(sentence_emotions: list) -> list:
    """
    Build the emotional journey by dividing the speech into sections.
    """
    if len(sentence_emotions) < 4:
        return [{'section': 'Speech', 'emotion': sentence_emotions[0]['dominant_emotion'] if sentence_emotions else 'neutral'}]
    
    n = len(sentence_emotions)
    sections = [
        ('Opening', sentence_emotions[:n//4]),
        ('Development', sentence_emotions[n//4:n//2]),
        ('Turning Point', sentence_emotions[n//2:3*n//4]),
        ('Conclusion', sentence_emotions[3*n//4:])
    ]
    
    journey = []
    for section_name, section_sents in sections:
        # Find dominant emotion in this section
        section_emotions = Counter()
        for s in section_sents:
            for emotion, count in s['emotions'].items():
                section_emotions[emotion] += count
        
        if section_emotions:
            dominant = section_emotions.most_common(1)[0][0]
        else:
            dominant = 'neutral'
        
        journey.append({
            'section': section_name,
            'emotion': dominant,
            'sentence_count': len(section_sents)
        })
    
    return journey


def _interpret_emotions(dominant: str, distribution: dict, journey: list) -> list:
    """Generate user-friendly emotion interpretation."""
    interps = []
    
    interps.append(f"Dominant Emotion: {dominant.capitalize()}")
    
    # Check for emotional variety
    active_emotions = [e for e, pct in distribution.items() if pct > 5]
    if len(active_emotions) >= 4:
        interps.append("Your speech has rich emotional variety, which keeps the audience engaged.")
    elif len(active_emotions) <= 2:
        interps.append("Consider incorporating more emotional variety to create a more dynamic speech.")
    
    # Emotional journey interpretation
    if len(journey) >= 4:
        journey_str = ' → '.join([f"{j['section']}: {j['emotion'].capitalize()}" for j in journey])
        interps.append(f"Emotional Journey: {journey_str}")
        
        # Check for emotional arc
        opening = journey[0]['emotion']
        conclusion = journey[-1]['emotion']
        
        if opening in ['fear', 'sadness'] and conclusion in ['joy', 'trust', 'anticipation']:
            interps.append("Your speech follows a powerful emotional arc from challenge to triumph — excellent storytelling!")
        elif opening == conclusion:
            interps.append("Your emotional tone is consistent throughout. Consider creating more emotional contrast.")
    
    return interps


def _empty_emotions():
    return {
        'dominant_emotion': 'Unknown',
        'emotion_counts': {}, 'distribution': {},
        'total_emotion_words': 0,
        'sentence_emotions': [], 'journey': [],
        'interpretation': ['No text for emotion analysis.'],
        'journey_labels': []
    }
