"""
Speech Evaluation Score Module for ToastMasterAI
Generates the final comprehensive speech evaluation score.

NOTE: This is an NLP-based academic evaluation, not an official Toastmasters judging system.
"""

from utils.helpers import EVALUATION_WEIGHTS


def compute_speech_score(analysis_results: dict, weights: dict = None) -> dict:
    """
    Compute the final speech evaluation score from all analysis results.
    
    Categories:
    - Content (from key message, TF-IDF)
    - Organization (from structure coherence, speech sections)
    - Vocabulary (from vocabulary richness, filler words)
    - Language (from complexity, language patterns)
    - Audience Engagement (from engagement analyzer)
    - Storytelling (from storytelling analyzer)
    - Emotional Impact (from emotion + sentiment)
    - Message Clarity (from key message, topic modeling)
    
    Args:
        analysis_results: Dictionary containing results from all analysis modules
        weights: Optional custom weights (defaults to EVALUATION_WEIGHTS)
    
    Returns:
        Dictionary with category scores, overall score, strengths, areas to improve
    """
    if weights is None:
        weights = EVALUATION_WEIGHTS
    
    # ─── Content Score (0-10) ─────────────────────────────────
    content_score = _compute_content_score(analysis_results)
    
    # ─── Organization Score (0-10) ────────────────────────────
    organization_score = _compute_organization_score(analysis_results)
    
    # ─── Vocabulary Score (0-10) ──────────────────────────────
    vocabulary_score = _compute_vocabulary_score(analysis_results)
    
    # ─── Language Score (0-10) ────────────────────────────────
    language_score = _compute_language_score(analysis_results)
    
    # ─── Audience Engagement Score (0-10) ─────────────────────
    engagement_score = _compute_engagement_eval_score(analysis_results)
    
    # ─── Storytelling Score (0-10) ────────────────────────────
    storytelling_score = _compute_storytelling_eval_score(analysis_results)
    
    # ─── Emotional Impact Score (0-10) ────────────────────────
    emotional_score = _compute_emotional_eval_score(analysis_results)
    
    # ─── Message Clarity Score (0-10) ─────────────────────────
    clarity_score = _compute_clarity_score(analysis_results)
    
    category_scores = {
        'content': round(content_score, 1),
        'organization': round(organization_score, 1),
        'vocabulary': round(vocabulary_score, 1),
        'language': round(language_score, 1),
        'audience_engagement': round(engagement_score, 1),
        'storytelling': round(storytelling_score, 1),
        'emotional_impact': round(emotional_score, 1),
        'message_clarity': round(clarity_score, 1)
    }
    
    # Overall weighted score (0-100)
    overall = sum(
        category_scores[cat] * weight * 10  # Multiply by 10 to convert 0-10 to 0-100 contribution
        for cat, weight in weights.items()
    )
    overall = round(min(100, max(0, overall)), 1)
    
    # Identify strengths and areas to improve
    strengths = []
    improvements = []
    
    sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    for cat, score in sorted_scores[:3]:
        if score >= 7:
            strengths.append(_category_strength(cat, score))
    
    for cat, score in sorted_scores[-3:]:
        if score < 7:
            improvements.append(_category_improvement(cat, score))
    
    # Generate personalized feedback
    feedback = generate_actionable_feedback(category_scores, analysis_results)
    
    return {
        'category_scores': category_scores,
        'overall_score': overall,
        'weights': weights,
        'strengths': strengths,
        'improvements': improvements,
        'feedback': feedback,
        'disclaimer': 'This is an NLP-based academic evaluation, not an official Toastmasters judging system.'
    }


def _compute_content_score(results: dict) -> float:
    """Score content based on word count, topic coverage, keyword richness."""
    score = 5.0  # Base score
    
    stats = results.get('statistics', {})
    tfidf = results.get('tfidf', {})
    
    word_count = stats.get('word_count', 0)
    if 300 <= word_count <= 1500:
        score += 2.0
    elif word_count > 100:
        score += 1.0
    
    # Keyword variety
    keywords = tfidf.get('keywords', [])
    if len(keywords) >= 10:
        score += 2.0
    elif len(keywords) >= 5:
        score += 1.0
    
    # Vocabulary richness
    richness = stats.get('vocabulary_richness', 0)
    if richness > 60:
        score += 1.0
    
    return min(10, score)


def _compute_organization_score(results: dict) -> float:
    """Score organization based on coherence and structure."""
    score = 5.0
    
    coherence = results.get('coherence', {})
    avg_coh = coherence.get('avg_coherence', 50)
    open_close = coherence.get('opening_conclusion_similarity', 50)
    
    if avg_coh > 60:
        score += 2.0
    elif avg_coh > 40:
        score += 1.0
    
    if open_close > 50:
        score += 2.0
    elif open_close > 30:
        score += 1.0
    
    # Sentence variation
    stats = results.get('statistics', {})
    std = stats.get('sentence_length_std', 0)
    if 3 <= std <= 10:
        score += 1.0
    
    return min(10, score)


def _compute_vocabulary_score(results: dict) -> float:
    """Score vocabulary based on richness, filler words, and diversity."""
    score = 5.0
    
    stats = results.get('statistics', {})
    richness = stats.get('vocabulary_richness', 0)
    
    if richness > 70:
        score += 3.0
    elif richness > 50:
        score += 2.0
    elif richness > 30:
        score += 1.0
    
    # Penalize excessive filler words
    fillers = stats.get('filler_words', {})
    total_fillers = sum(fillers.values())
    if total_fillers <= 2:
        score += 1.0
    elif total_fillers > 10:
        score -= 1.0
    
    # Overused words penalty
    overused = stats.get('overused_words', {})
    if len(overused) <= 1:
        score += 1.0
    elif len(overused) > 5:
        score -= 1.0
    
    return min(10, max(0, score))


def _compute_language_score(results: dict) -> float:
    """Score language quality based on complexity and patterns."""
    score = 5.0
    
    complexity = results.get('complexity', {})
    comp_score = complexity.get('overall_score', 50)
    
    # Moderate complexity is ideal
    if 40 <= comp_score <= 70:
        score += 3.0
    elif 30 <= comp_score <= 80:
        score += 2.0
    else:
        score += 1.0
    
    language_pattern = results.get('language_model', {})
    lp_score = language_pattern.get('score', 50)
    if lp_score > 60:
        score += 2.0
    elif lp_score > 40:
        score += 1.0
    
    return min(10, score)


def _compute_engagement_eval_score(results: dict) -> float:
    """Score audience engagement."""
    engagement = results.get('engagement', {})
    eng_score = engagement.get('score', 0)
    
    # Map 0-100 to 0-10
    return min(10, eng_score / 10)


def _compute_storytelling_eval_score(results: dict) -> float:
    """Score storytelling quality."""
    storytelling = results.get('storytelling', {})
    story_score = storytelling.get('story_score', 0)
    
    return min(10, story_score / 10)


def _compute_emotional_eval_score(results: dict) -> float:
    """Score emotional impact."""
    score = 5.0
    
    emotion = results.get('emotion', {})
    sentiment = results.get('sentiment', {})
    
    # Emotional variety
    distribution = emotion.get('distribution', {})
    active_emotions = sum(1 for v in distribution.values() if v > 5)
    if active_emotions >= 4:
        score += 3.0
    elif active_emotions >= 2:
        score += 1.5
    
    # Sentiment trajectory
    timeline = sentiment.get('timeline', [])
    if len(timeline) >= 4:
        start = sum(timeline[:len(timeline)//4]) / max(1, len(timeline)//4)
        end = sum(timeline[-(len(timeline)//4):]) / max(1, len(timeline)//4)
        if end > start:  # Positive trajectory
            score += 2.0
    
    return min(10, score)


def _compute_clarity_score(results: dict) -> float:
    """Score message clarity."""
    score = 5.0
    
    tfidf = results.get('tfidf', {})
    topic = results.get('topic_modeling', {})
    
    # Clear keywords
    keywords = tfidf.get('keywords', [])
    if keywords:
        top_score = keywords[0].get('score', 0)
        if top_score > 0.1:
            score += 2.0
    
    # Clear primary topic
    primary = topic.get('primary_topic', '')
    if primary and primary != 'Unknown':
        score += 2.0
    
    # Topic concentration
    dist = topic.get('topic_distribution', {})
    if dist:
        max_topic_pct = max(dist.values()) if dist else 0
        if max_topic_pct > 50:
            score += 1.0
    
    return min(10, score)


def _category_strength(category: str, score: float) -> str:
    """Generate a strength statement for a category."""
    labels = {
        'content': 'Strong content coverage',
        'organization': 'Well-organized speech structure',
        'vocabulary': 'Excellent vocabulary diversity',
        'language': 'Strong language quality',
        'audience_engagement': 'Effective audience engagement',
        'storytelling': 'Compelling storytelling',
        'emotional_impact': 'Powerful emotional impact',
        'message_clarity': 'Clear central message'
    }
    return labels.get(category, f'Strong {category}') + f' ({score}/10)'


def _category_improvement(category: str, score: float) -> str:
    """Generate an improvement suggestion for a category."""
    labels = {
        'content': 'Enrich content with more detailed examples or evidence',
        'organization': 'Improve speech structure with clearer transitions between sections',
        'vocabulary': 'Diversify vocabulary and reduce filler or overused words',
        'language': 'Adjust language complexity for your target audience',
        'audience_engagement': 'Add more questions, direct references, or calls to action',
        'storytelling': 'Include more personal stories with characters and specific details',
        'emotional_impact': 'Create more emotional variety throughout your speech',
        'message_clarity': 'Make your central message more explicit and repeat it strategically'
    }
    return labels.get(category, f'Improve {category}') + f' (current: {score}/10)'


def generate_actionable_feedback(scores: dict, results: dict) -> list:
    """
    Generate 3-5 personalized, actionable recommendations based on actual analysis.
    """
    feedback = []
    
    # Check repetition
    repetition = results.get('repetition', {})
    rep_score = repetition.get('repetition_score', 0)
    if rep_score > 40:
        top_phrases = repetition.get('repeated_phrases', [])
        if top_phrases:
            feedback.append(
                f'Reduce repeated phrases. "{top_phrases[0]["phrase"]}" appears {top_phrases[0]["count"]} times. '
                f'Consider varying your phrasing for a more polished delivery.'
            )
    
    # Check sentence length
    stats = results.get('statistics', {})
    avg_sent = stats.get('avg_sentence_length', 0)
    if avg_sent > 25:
        feedback.append(
            f'Your average sentence length is {avg_sent:.0f} words. '
            f'Shorten some sentences to improve clarity and create natural pauses.'
        )
    
    # Check engagement
    engagement = results.get('engagement', {})
    if engagement.get('questions', 0) < 2:
        feedback.append(
            'Add more audience-directed questions to increase engagement and interaction.'
        )
    
    # Check emotional arc
    emotion = results.get('emotion', {})
    journey = emotion.get('journey', [])
    if len(journey) >= 4:
        if journey[-1].get('emotion') in ['fear', 'sadness', 'anger']:
            feedback.append(
                'Your speech ends on a heavy emotional note. Consider closing with hope or confidence for a more uplifting conclusion.'
            )
    
    # Check vocabulary
    if scores.get('vocabulary', 0) < 7:
        fillers = stats.get('filler_words', {})
        if fillers:
            top_filler = max(fillers, key=fillers.get)
            feedback.append(
                f'Reduce usage of filler words. You used "{top_filler}" {fillers[top_filler]} times.'
            )
    
    # Check storytelling
    storytelling = results.get('storytelling', {})
    if storytelling.get('story_score', 0) < 40:
        feedback.append(
            'Strengthen storytelling by adding specific characters, locations, and time references.'
        )
    
    # Check coherence
    coherence = results.get('coherence', {})
    if coherence.get('opening_conclusion_similarity', 100) < 40:
        feedback.append(
            'Strengthen the connection between your opening and conclusion — reinforce your central message at the end.'
        )
    
    # Positive feedback
    sentiment = results.get('sentiment', {})
    if sentiment.get('overall_sentiment') == 'Positive':
        feedback.append(
            'Your speech maintains a positive tone throughout. This works well for motivational and inspirational speeches.'
        )
    
    if engagement.get('score', 0) > 70:
        feedback.append(
            'Excellent audience engagement! Your use of questions and direct references keeps the audience involved.'
        )
    
    # Return 3-5 most relevant
    return feedback[:5]
