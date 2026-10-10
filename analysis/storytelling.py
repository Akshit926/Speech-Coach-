"""
Storytelling Analyzer Module for ToastMasterAI
Detects storytelling elements: characters, places, events, emotional statements.
"""

from analysis.ner import extract_entities
from analysis.sentiment import analyze_sentiment
from preprocessing.tokenizer import tokenize_sentences, get_word_tokens_no_punct
import re


# Story-related patterns
STORY_MARKERS = [
    r'\bi remember\b', r'\bone day\b', r'\bthat day\b', r'\bthat moment\b',
    r'\bi was\b', r'\bwe were\b', r'\bhe was\b', r'\bshe was\b',
    r'\bi realized\b', r'\bi learned\b', r'\bi discovered\b',
    r'\bonce upon\b', r'\byears ago\b', r'\bmonths ago\b',
    r'\blet me tell you\b', r'\bi want to tell\b', r'\bi want to share\b',
    r'\bthat night\b', r'\bthat morning\b', r'\bthat evening\b',
    r'\bfirst time\b', r'\blast time\b', r'\bnext day\b',
    r'\bturning point\b', r'\beverything changed\b', r'\bsomething happened\b',
    r'\bwhen i was\b', r'\bgrowing up\b', r'\bas a child\b'
]

# Action verbs indicating story progression
ACTION_VERBS = {
    'walked', 'ran', 'said', 'told', 'asked', 'went', 'came', 'saw',
    'found', 'felt', 'thought', 'decided', 'started', 'stopped',
    'tried', 'failed', 'succeeded', 'changed', 'moved', 'left',
    'arrived', 'returned', 'looked', 'smiled', 'cried', 'laughed',
    'stood', 'sat', 'spoke', 'heard', 'noticed', 'remembered'
}

# Problem/conflict indicators
PROBLEM_MARKERS = [
    r'\bproblem\b', r'\bchallenge\b', r'\bstruggle\b', r'\bdifficult\b',
    r'\bfailed\b', r'\bfailure\b', r'\bfear\b', r'\bafraid\b',
    r'\bworried\b', r'\bstuck\b', r'\bconfused\b', r'\blost\b',
    r'\bmistake\b', r'\bwrong\b', r'\bcrisis\b', r'\bdisaster\b',
    r'\bnightmare\b', r'\bworst\b', r'\bdevastated\b', r'\bterrible\b'
]

# Resolution indicators
RESOLUTION_MARKERS = [
    r'\brealized\b', r'\blearned\b', r'\bunderstood\b', r'\bdiscovered\b',
    r'\btransformed\b', r'\bchanged\b', r'\bimproved\b', r'\bgrew\b',
    r'\bovercome\b', r'\bsucceeded\b', r'\bachieved\b', r'\bsolved\b',
    r'\btoday\b.*\bbetter\b', r'\bnow\b.*\bknow\b', r'\bfinally\b'
]


def analyze_storytelling(text: str, ner_results: dict = None, sentiment_data: dict = None) -> dict:
    """
    Analyze storytelling elements in the speech.
    
    Returns:
        Dictionary with character count, locations, events, story score, etc.
    """
    if not text:
        return _empty_storytelling()
    
    # Get NER results
    if ner_results is None:
        ner_results = extract_entities(text)
    
    sentences = tokenize_sentences(text)
    text_lower = text.lower()
    
    # Count story-related sentences
    story_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        is_story = False
        
        # Check for story markers
        for pattern in STORY_MARKERS:
            if re.search(pattern, sent_lower):
                is_story = True
                break
        
        # Check for action verbs
        words = set(get_word_tokens_no_punct(sent_lower))
        if len(words & ACTION_VERBS) >= 2:
            is_story = True
        
        if is_story:
            story_sentences.append(sent)
    
    # Count problems/conflicts
    problem_sentences = []
    for sent in sentences:
        for pattern in PROBLEM_MARKERS:
            if re.search(pattern, sent.lower()):
                problem_sentences.append(sent)
                break
    
    # Count resolutions
    resolution_sentences = []
    for sent in sentences:
        for pattern in RESOLUTION_MARKERS:
            if re.search(pattern, sent.lower()):
                resolution_sentences.append(sent)
                break
    
    # Detect emotional statements (sentences with strong sentiment)
    if sentiment_data is None:
        sentiment_data = analyze_sentiment(text)
    emotional_sentences = [
        s for s in sentiment_data.get('sentence_sentiments', [])
        if abs(s.get('compound', 0)) > 0.5
    ]
    
    # Story elements summary
    characters = len(ner_results.get('people', []))
    locations = len(ner_results.get('locations', []))
    dates = len(ner_results.get('dates', []))
    events = len(ner_results.get('events', []))
    
    # Story score (0-100)
    story_score = _compute_story_score(
        characters, locations, dates, 
        len(story_sentences), len(problem_sentences), 
        len(resolution_sentences), len(emotional_sentences),
        len(sentences)
    )
    
    # Generate interpretation
    interpretation = _interpret_storytelling(
        characters, locations, dates, events,
        len(story_sentences), len(problem_sentences),
        len(resolution_sentences), len(emotional_sentences),
        story_score, len(sentences)
    )
    
    return {
        'characters': ner_results.get('people', []),
        'character_count': characters,
        'locations': ner_results.get('locations', []),
        'location_count': locations,
        'dates': ner_results.get('dates', []),
        'date_count': dates,
        'events': ner_results.get('events', []),
        'event_count': events,
        'story_sentences_count': len(story_sentences),
        'problem_sentences_count': len(problem_sentences),
        'resolution_sentences_count': len(resolution_sentences),
        'emotional_sentences_count': len(emotional_sentences),
        'total_sentences': len(sentences),
        'story_score': story_score,
        'interpretation': interpretation,
        'has_conflict': len(problem_sentences) > 0,
        'has_resolution': len(resolution_sentences) > 0,
        'has_characters': characters > 0,
        'story_sentences': story_sentences[:5]  # Sample
    }


def _compute_story_score(characters, locations, dates, story_sents,
                          problems, resolutions, emotional, total_sents):
    """Compute a storytelling score from 0-100."""
    if total_sents == 0:
        return 0
    
    score = 0
    
    # Characters (max 20 points)
    score += min(20, characters * 7)
    
    # Locations (max 10 points)
    score += min(10, locations * 5)
    
    # Time references (max 10 points)
    score += min(10, dates * 3)
    
    # Story sentence ratio (max 25 points)
    story_ratio = story_sents / total_sents
    score += min(25, int(story_ratio * 50))
    
    # Conflict presence (max 10 points)
    score += min(10, problems * 5)
    
    # Resolution presence (max 10 points)
    score += min(10, resolutions * 5)
    
    # Emotional content (max 15 points)
    emotion_ratio = emotional / total_sents if total_sents > 0 else 0
    score += min(15, int(emotion_ratio * 30))
    
    return min(100, score)


def _interpret_storytelling(characters, locations, dates, events,
                             story_sents, problems, resolutions,
                             emotional, score, total_sents):
    """Generate human-readable storytelling interpretation."""
    interps = []
    
    if score >= 70:
        interps.append("Your speech contains strong storytelling elements with identifiable characters, settings, and emotional progression.")
    elif score >= 40:
        interps.append("Your speech includes some storytelling elements. Consider adding more vivid characters and specific details.")
    else:
        interps.append("Your speech has minimal storytelling. Adding personal stories or anecdotes could significantly improve engagement.")
    
    if characters > 0:
        interps.append(f"Characters detected: {characters}. Named characters help the audience connect with your story.")
    
    if problems > 0 and resolutions > 0:
        interps.append("Your speech presents a clear problem and resolution — a hallmark of effective storytelling.")
    elif problems > 0 and resolutions == 0:
        interps.append("You present challenges but the resolution could be clearer. Consider making the lesson or outcome more explicit.")
    
    if emotional > 3:
        interps.append("Your speech has strong emotional content, which enhances the storytelling impact.")
    
    story_ratio = (story_sents / total_sents * 100) if total_sents > 0 else 0
    interps.append(f"Story-related sentences: {story_sents} out of {total_sents} ({story_ratio:.0f}%)")
    
    return interps


def _empty_storytelling():
    return {
        'characters': [], 'character_count': 0,
        'locations': [], 'location_count': 0,
        'dates': [], 'date_count': 0,
        'events': [], 'event_count': 0,
        'story_sentences_count': 0, 'problem_sentences_count': 0,
        'resolution_sentences_count': 0, 'emotional_sentences_count': 0,
        'total_sentences': 0, 'story_score': 0,
        'interpretation': ['No text for storytelling analysis.'],
        'has_conflict': False, 'has_resolution': False, 'has_characters': False,
        'story_sentences': []
    }
