"""
Utility Helpers for ToastMasterAI
Shared constants, configurations, and helper functions.
"""

import os
import csv
import pandas as pd

# ─── Configurable Constants ───────────────────────────────────────

# Speaking speed options (words per minute)
SPEAKING_SPEEDS = {
    100: "Slow & deliberate",
    120: "Conversational",
    140: "Standard Toastmasters",
    160: "Fast-paced"
}

DEFAULT_SPEAKING_SPEED = 140

# Filler/overused words list (configurable)
FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
    'literally', 'honestly', 'right', 'well', 'okay', 'just',
    'stuff', 'things', 'kind of', 'sort of', 'i mean', 'anyway'
]

OVERUSED_WORDS = [
    'very', 'really', 'good', 'nice', 'great', 'bad', 'big',
    'small', 'thing', 'stuff', 'got', 'get', 'said', 'went',
    'lot', 'pretty', 'quite', 'amazing', 'awesome', 'incredible'
]

# Speech categories for classification
SPEECH_CATEGORIES = [
    'Personal Development', 'Leadership', 'Communication',
    'Persuasive', 'Educational', 'Humor',
    'Experience Sharing', 'Teamwork', 'Motivation'
]

# Evaluation weights (configurable)
EVALUATION_WEIGHTS = {
    'content': 0.15,
    'organization': 0.12,
    'vocabulary': 0.12,
    'language': 0.10,
    'audience_engagement': 0.13,
    'storytelling': 0.13,
    'emotional_impact': 0.12,
    'message_clarity': 0.13
}

# Complexity score interpretation
COMPLEXITY_LEVELS = {
    (0, 30): "Very Simple",
    (31, 50): "Simple",
    (51, 70): "Moderate",
    (71, 85): "Advanced",
    (86, 100): "Highly Complex"
}

# ─── Helper Functions ─────────────────────────────────────────────

def estimate_speaking_time(word_count: int, wpm: int = DEFAULT_SPEAKING_SPEED) -> str:
    """
    Estimate speaking time from word count and speaking speed.
    
    Returns:
        Formatted time string like '5 min 18 sec'
    """
    if word_count <= 0 or wpm <= 0:
        return "0 min 0 sec"
    
    total_seconds = int((word_count / wpm) * 60)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes} min {seconds} sec"


def estimate_speaking_time_seconds(word_count: int, wpm: int = DEFAULT_SPEAKING_SPEED) -> int:
    """Return estimated speaking time in seconds."""
    if word_count <= 0 or wpm <= 0:
        return 0
    return int((word_count / wpm) * 60)


def get_complexity_label(score: float) -> str:
    """Get human-readable complexity label from score."""
    for (low, high), label in COMPLEXITY_LEVELS.items():
        if low <= score <= high:
            return label
    return "Unknown"


def load_speeches_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the speeches dataset.
    
    Args:
        filepath: Path to CSV file. Defaults to data/speeches.csv.
    
    Returns:
        pandas DataFrame
    """
    if filepath is None:
        # Try to find the data file relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'speeches.csv')
    
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['speech_id', 'title', 'speaker', 'category', 'text'])
    
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame(columns=['speech_id', 'title', 'speaker', 'category', 'text'])


def save_speeches_dataset(df: pd.DataFrame, filepath: str = None):
    """Save the speeches dataset to CSV."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'speeches.csv')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def format_percentage(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:.1f}%"


def format_score(value: float, max_val: float = 10.0) -> str:
    """Format a score with its maximum."""
    return f"{value:.1f}/{max_val:.0f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_time(seconds: int) -> str:
    """Format total seconds into 'X min Y sec' format."""
    if not seconds or seconds <= 0:
        return "0 min 0 sec"
    minutes = int(seconds) // 60
    rem_seconds = int(seconds) % 60
    return f"{minutes} min {rem_seconds} sec"


def get_score_badge(score: float) -> str:
    """Get a Toastmasters-styled honor badge based on overall score."""
    if score >= 90:
        return "Distinguished Speaker (Gold)"
    elif score >= 80:
        return "Competent Communicator (Silver)"
    elif score >= 70:
        return "Proficient Speaker (Bronze)"
    elif score >= 60:
        return "Emerging Speaker"
    else:
        return "Developing Speaker"
