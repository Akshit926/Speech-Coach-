"""
Named Entity Recognition Module for ToastMasterAI
Detects people, organizations, locations, dates, events using spaCy.
"""

import spacy
from collections import Counter, defaultdict


def _load_spacy_model():
    """Load spaCy English model with fallback."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        # Try downloading
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], 
                       capture_output=True, check=False)
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            return None


# Lazy-load spaCy model
_nlp = None

def get_nlp():
    """Get or load the spaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp


# Entity label descriptions for user-friendly display
ENTITY_DESCRIPTIONS = {
    'PERSON': 'People/Characters',
    'ORG': 'Organizations',
    'GPE': 'Countries/Cities/States',
    'LOC': 'Locations',
    'DATE': 'Dates/Time References',
    'TIME': 'Time References',
    'EVENT': 'Events',
    'FAC': 'Facilities/Buildings',
    'NORP': 'Groups/Nationalities',
    'MONEY': 'Monetary Values',
    'PERCENT': 'Percentages',
    'CARDINAL': 'Numbers',
    'ORDINAL': 'Ordinal Numbers',
    'QUANTITY': 'Quantities',
    'WORK_OF_ART': 'Works of Art',
    'PRODUCT': 'Products',
    'LANGUAGE': 'Languages'
}


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary with entity details, counts, and interpretation
    """
    nlp = get_nlp()
    
    if nlp is None or not text:
        return _empty_ner()
    
    try:
        doc = nlp(text)
    except Exception:
        return _empty_ner()
    
    # Collect entities
    entities = []
    entity_counts = Counter()
    entity_by_type = defaultdict(list)
    
    for ent in doc.ents:
        entity_info = {
            'text': ent.text,
            'label': ent.label_,
            'description': ENTITY_DESCRIPTIONS.get(ent.label_, ent.label_),
            'start': ent.start_char,
            'end': ent.end_char
        }
        entities.append(entity_info)
        entity_counts[ent.label_] += 1
        
        # Avoid duplicates in the type list
        if ent.text not in entity_by_type[ent.label_]:
            entity_by_type[ent.label_].append(ent.text)
    
    # Create summary counts
    summary = {}
    for label, count in entity_counts.items():
        friendly_name = ENTITY_DESCRIPTIONS.get(label, label)
        summary[friendly_name] = count
    
    return {
        'entities': entities,
        'entity_counts': dict(entity_counts),
        'entity_by_type': dict(entity_by_type),
        'summary': summary,
        'total_entities': len(entities),
        'unique_entities': sum(len(v) for v in entity_by_type.values()),
        'people': entity_by_type.get('PERSON', []),
        'organizations': entity_by_type.get('ORG', []),
        'locations': entity_by_type.get('GPE', []) + entity_by_type.get('LOC', []),
        'dates': entity_by_type.get('DATE', []) + entity_by_type.get('TIME', []),
        'events': entity_by_type.get('EVENT', [])
    }


def _empty_ner():
    """Return empty NER result."""
    return {
        'entities': [], 'entity_counts': {}, 'entity_by_type': {},
        'summary': {}, 'total_entities': 0, 'unique_entities': 0,
        'people': [], 'organizations': [], 'locations': [],
        'dates': [], 'events': []
    }
