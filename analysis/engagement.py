"""
Audience Engagement Analyzer Module for ToastMasterAI
Detects questions, rhetorical questions, direct references, calls to action.
"""

import re
from preprocessing.tokenizer import tokenize_sentences


# Patterns for audience engagement elements
QUESTION_PATTERN = re.compile(r'[^.]*\?')

RHETORICAL_MARKERS = [
    r'\bhave you ever\b', r'\bdo you\b', r'\bcan you\b', r'\bwould you\b',
    r'\bhow many\b', r'\bhow often\b', r'\bwhen was\b', r'\bwhat if\b',
    r'\bisn\'?t it\b', r'\bwouldn\'?t you\b', r'\bright\?\b',
    r'\bwho here\b', r'\braise your hand\b'
]

AUDIENCE_REFERENCE_PATTERNS = [
    r'\byou\b', r'\byour\b', r'\byourself\b', r'\byourselves\b',
    r'\beveryone\b', r'\beverybody\b', r'\bfellow\b', r'\bfriends\b',
    r'\bladies and gentlemen\b', r'\btoastmasters\b', r'\bguests\b',
    r'\bwe\b', r'\bour\b', r'\bus\b', r'\btogether\b',
    r'\bmy friends\b', r'\bfolks\b'
]

CALL_TO_ACTION_PATTERNS = [
    r'\bi challenge you\b', r'\bi encourage\b', r'\bi invite you\b',
    r'\bi ask you\b', r'\bi want you to\b', r'\bi urge you\b',
    r'\blet us\b', r"\blet's\b", r'\btry\b.*\btoday\b',
    r'\bstart\b.*\btoday\b', r'\bgo ahead\b', r'\btake action\b',
    r'\bremember\b', r'\bthink about\b', r'\bconsider\b',
    r'\bmy challenge\b', r'\bmy request\b'
]

INTERACTIVE_PATTERNS = [
    r'\bclose your eyes\b', r'\braise your hand\b', r'\blook around\b',
    r'\bimagine\b', r'\bpicture\b', r'\bthink of\b',
    r'\btry something\b', r'\btry this\b', r'\bgo ahead\b'
]


def analyze_engagement(text: str) -> dict:
    """
    Analyze audience engagement elements in the speech.
    
    Returns:
        Dictionary with engagement metrics, examples, and score
    """
    if not text:
        return _empty_engagement()
    
    sentences = tokenize_sentences(text)
    text_lower = text.lower()
    
    # Questions
    questions = [s for s in sentences if '?' in s]
    
    # Rhetorical questions
    rhetorical = []
    for sent in questions:
        for pattern in RHETORICAL_MARKERS:
            if re.search(pattern, sent.lower()):
                rhetorical.append(sent)
                break
    
    # Direct audience references
    audience_refs = []
    for sent in sentences:
        for pattern in AUDIENCE_REFERENCE_PATTERNS:
            if re.search(pattern, sent.lower()):
                audience_refs.append(sent)
                break
    
    # Calls to action
    calls_to_action = []
    for sent in sentences:
        for pattern in CALL_TO_ACTION_PATTERNS:
            if re.search(pattern, sent.lower()):
                calls_to_action.append(sent)
                break
    
    # Interactive statements
    interactive = []
    for sent in sentences:
        for pattern in INTERACTIVE_PATTERNS:
            if re.search(pattern, sent.lower()):
                interactive.append(sent)
                break
    
    # Emotional/impactful statements (sentences ending with !)
    emotional_statements = [s for s in sentences if s.strip().endswith('!')]
    
    # Calculate engagement score (0-100)
    score = _compute_engagement_score(
        len(questions), len(rhetorical), len(audience_refs),
        len(calls_to_action), len(interactive), len(emotional_statements),
        len(sentences)
    )
    
    # Determine engagement level
    if score >= 75:
        level = "High"
    elif score >= 50:
        level = "Moderate"
    elif score >= 25:
        level = "Low"
    else:
        level = "Very Low"
    
    # Generate suggestions
    suggestions = _generate_engagement_suggestions(
        len(questions), len(audience_refs), len(calls_to_action),
        len(interactive), score, len(sentences)
    )
    
    return {
        'score': score,
        'level': level,
        'questions': len(questions),
        'rhetorical_questions': len(rhetorical),
        'audience_references': len(set(audience_refs)),
        'calls_to_action': len(calls_to_action),
        'interactive_elements': len(interactive),
        'emotional_statements': len(emotional_statements),
        'total_sentences': len(sentences),
        'question_examples': questions[:3],
        'cta_examples': calls_to_action[:3],
        'interactive_examples': interactive[:3],
        'suggestions': suggestions
    }


def _compute_engagement_score(questions, rhetorical, audience_refs,
                               ctas, interactive, emotional, total):
    """Compute audience engagement score from 0-100."""
    if total == 0:
        return 0
    
    score = 0
    
    # Questions (max 20 points)
    q_ratio = questions / total
    score += min(20, int(q_ratio * 100))
    
    # Rhetorical questions bonus (max 10 points)
    score += min(10, rhetorical * 5)
    
    # Audience references (max 25 points)
    ref_ratio = audience_refs / total
    score += min(25, int(ref_ratio * 50))
    
    # Calls to action (max 20 points)
    score += min(20, ctas * 7)
    
    # Interactive elements (max 15 points)
    score += min(15, interactive * 8)
    
    # Emotional statements (max 10 points)
    score += min(10, emotional * 3)
    
    return min(100, score)


def _generate_engagement_suggestions(questions, audience_refs, ctas,
                                      interactive, score, total):
    """Generate actionable engagement improvement suggestions."""
    suggestions = []
    
    if questions == 0:
        suggestions.append("Consider adding questions to your speech to involve the audience actively.")
    elif questions < 3:
        suggestions.append("Adding a few more questions could further boost audience interaction.")
    
    if audience_refs < total * 0.1:
        suggestions.append("Use more direct audience references ('you', 'we', 'your') to create a personal connection.")
    
    if ctas == 0:
        suggestions.append("Include a clear call to action — tell the audience what you want them to do or think differently.")
    
    if interactive == 0:
        suggestions.append("Consider adding interactive elements like 'imagine...', 'think about...', or 'raise your hand if...'.")
    
    if score >= 75:
        suggestions.append("Excellent audience engagement! Your speech actively involves the listener.")
    
    return suggestions


def _empty_engagement():
    return {
        'score': 0, 'level': 'Unknown',
        'questions': 0, 'rhetorical_questions': 0,
        'audience_references': 0, 'calls_to_action': 0,
        'interactive_elements': 0, 'emotional_statements': 0,
        'total_sentences': 0,
        'question_examples': [], 'cta_examples': [], 'interactive_examples': [],
        'suggestions': ['No text for engagement analysis.']
    }
