"""
Text Cleaning Module for ToastMasterAI
Handles lowercasing, punctuation removal, whitespace normalization.
"""

import re
import unicodedata


def clean_text(text: str, lowercase: bool = True, remove_punctuation: bool = False,
               normalize_whitespace: bool = True) -> str:
    """
    Clean input text while preserving sentence structure.
    
    Args:
        text: Raw input text
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation (use cautiously)
        normalize_whitespace: Whether to normalize extra whitespace
    
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation (optional — only for certain analyses)
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    return text


def remove_special_characters(text: str) -> str:
    """Remove special characters but keep basic punctuation for sentence detection."""
    if not text:
        return ""
    # Keep periods, commas, question marks, exclamation marks
    text = re.sub(r'[^\w\s.,?!;:\'\"-]', '', text)
    return text


def normalize_quotes(text: str) -> str:
    """Normalize different quote styles to standard quotes."""
    if not text:
        return ""
    # Replace smart quotes with standard quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def preprocess_for_analysis(text: str) -> dict:
    """
    Run the full cleaning pipeline and return both original and cleaned versions.
    
    Returns:
        dict with 'original', 'cleaned', 'cleaned_no_punct' keys
    """
    if not text or not isinstance(text, str):
        return {
            'original': '',
            'cleaned': '',
            'cleaned_no_punct': ''
        }
    
    normalized = normalize_quotes(text.strip())
    cleaned = clean_text(normalized, lowercase=True, remove_punctuation=False)
    cleaned_no_punct = clean_text(normalized, lowercase=True, remove_punctuation=True)
    
    return {
        'original': normalized,
        'cleaned': cleaned,
        'cleaned_no_punct': cleaned_no_punct
    }
