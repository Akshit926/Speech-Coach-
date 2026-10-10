"""
Named Entity Recognition Module for ToastMasterAI
Detects people, organizations, locations, dates, events using spaCy.
"""

try:
    import spacy
except ImportError:
    spacy = None

import re
from collections import Counter, defaultdict
import nltk
from preprocessing.tokenizer import tokenize_sentences, tokenize_words


def _load_spacy_model():
    """Load spaCy English model with fallback."""
    if spacy is None:
        return None
    try:
        return spacy.load('en_core_web_sm')
    except Exception:
        return None


# Lazy-load spaCy model
_nlp = None

def get_nlp():
    """Get or load the spaCy NLP model."""
    global _nlp
    if _nlp is None and spacy is not None:
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
    if not text:
        return _empty_ner()

    nlp = get_nlp()
    
    if nlp is None:
        return _extract_entities_nltk(text)
    
    try:
        doc = nlp(text)
    except Exception:
        return _extract_entities_nltk(text)
    
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


def _extract_entities_nltk(text: str) -> dict:
    """Fallback entity extraction using NLTK and regex patterns."""
    if not text:
        return _empty_ner()
            
    entities = []
    entity_counts = Counter()
    entity_by_type = defaultdict(list)
    
    # Try NLTK ne_chunk if chunker is available
    has_ne_chunk = False
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
        has_ne_chunk = True
    except LookupError:
        pass

    try:
        sentences = tokenize_sentences(text)
        for sent in sentences[:25]:  # Process representative sample of sentences for high performance
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            
            if has_ne_chunk:
                try:
                    tree = nltk.ne_chunk(tagged)
                    for subtree in tree:
                        if hasattr(subtree, 'label'):
                            ent_label = subtree.label()
                            ent_text = " ".join([token for token, pos in subtree.leaves()])
                            
                            label_map = {
                                'ORGANIZATION': 'ORG',
                                'PERSON': 'PERSON',
                                'GPE': 'GPE',
                                'LOCATION': 'LOC',
                                'FACILITY': 'FAC'
                            }
                            mapped_label = label_map.get(ent_label, ent_label)
                            
                            entities.append({
                                'text': ent_text,
                                'label': mapped_label,
                                'description': ENTITY_DESCRIPTIONS.get(mapped_label, mapped_label),
                                'start': text.find(ent_text),
                                'end': text.find(ent_text) + len(ent_text)
                            })
                            entity_counts[mapped_label] += 1
                            if ent_text not in entity_by_type[mapped_label]:
                                entity_by_type[mapped_label].append(ent_text)
                    continue
                except Exception:
                    pass
            
            # Fast POS-based proper noun extraction fallback
            for token, pos in tagged:
                if pos in ('NNP', 'NNPS') and len(token) > 2 and token.isalpha():
                    label = 'PERSON'
                    entities.append({
                        'text': token,
                        'label': label,
                        'description': ENTITY_DESCRIPTIONS.get(label, 'People/Characters'),
                        'start': text.find(token),
                        'end': text.find(token) + len(token)
                    })
                    entity_counts[label] += 1
                    if token not in entity_by_type[label]:
                        entity_by_type[label].append(token)
    except Exception:
        pass
    # Regex patterns for dates and numbers/percentages
    date_patterns = re.findall(r'\b(?:(?:19|20)\d{2}|January|February|March|April|May|June|July|August|September|October|November|December)\b', text, re.IGNORECASE)
    for d in set(date_patterns):
        if d not in entity_by_type['DATE']:
            entity_by_type['DATE'].append(d)
            entity_counts['DATE'] += 1
            entities.append({
                'text': d,
                'label': 'DATE',
                'description': ENTITY_DESCRIPTIONS.get('DATE', 'Dates/Time References'),
                'start': text.find(d),
                'end': text.find(d) + len(d)
            })

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
