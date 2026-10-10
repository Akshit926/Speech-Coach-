"""
ToastMasterAI: An Intelligent NLP-Based Toastmasters Speech Analysis and Coaching System
Academic Mini-Project for T.Y. B.Tech CSE (AI & ML), NLP Subject, PCCOE, Pune

Author: Akshit & Team
"""

import os
import sys
import time
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Preprocessing & Utilities
from utils.helpers import (
    load_speeches_dataset,
    format_time,
    get_score_badge,
    EVALUATION_WEIGHTS,
    SPEECH_CATEGORIES,
    FILLER_WORDS
)
from preprocessing.cleaner import clean_text
from preprocessing.tokenizer import tokenize_sentences, tokenize_words

# Import Analysis Modules
from analysis.statistics import compute_statistics, get_vocabulary_coach_data
from analysis.language_detection import detect_language
from analysis.pos_analysis import analyze_pos
from analysis.ner import extract_entities
from analysis.tfidf import extract_keywords_tfidf
from analysis.embeddings import analyze_section_coherence, compute_speech_similarity
from analysis.ngrams import analyze_ngrams
from analysis.language_model import compute_language_pattern_score
from analysis.topic_modeling import discover_topics
from analysis.sentiment import analyze_sentiment
from analysis.emotion import detect_emotions
from analysis.storytelling import analyze_storytelling
from analysis.engagement import analyze_engagement
from analysis.repetition import analyze_repetition
from analysis.complexity import compute_complexity

# Import Classification & Evaluation
from classification.speech_classifier import SpeechClassifier
from evaluation.speech_score import compute_speech_score
from scraping.scraper import scrape_page, clean_scraped_text

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToastMasterAI — AI Speech Coach",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Executive Styling ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(212, 175, 55, 0.15) 0%, rgba(212, 175, 55, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    .academic-pill {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #F59E0B;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 50px;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-sub {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    .score-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(#F59E0B var(--score-deg), #E2E8F0 0deg);
        position: relative;
    }

    .feedback-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.8rem;
    }

    .strength-box {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        color: #065F46;
        font-weight: 500;
    }

    .improve-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        color: #92400E;
        font-weight: 500;
    }

    .badge-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    .disclaimer-banner {
        background: #F1F5F9;
        border: 1px dashed #CBD5E1;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Caching & Initialization ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_classifier(model_type='logistic_regression'):
    """Load or train speech classifier."""
    clf = SpeechClassifier(model_type=model_type)
    model_filenames = {
        'logistic_regression': 'speech_classifier.pkl',
        'naive_bayes': 'speech_classifier_nb.pkl',
        'svm': 'speech_classifier_svm.pkl'
    }
    filename = model_filenames.get(model_type, 'speech_classifier.pkl')
    loaded = clf.load_model(filename)
    if not loaded:
        df = load_speeches_dataset()
        if not df.empty:
            clf.train(df['text'].tolist(), df['category'].tolist())
    return clf


@st.cache_data(show_spinner=False)
def run_nlp_analysis(text: str, wpm: int = 140):
    """
    Run the complete 18-module NLP analysis pipeline on input speech text.
    """
    cleaned = clean_text(text)
    
    # 1. Statistics
    stats = compute_statistics(text, wpm=wpm)
    
    # 2. Language Detection
    lang = detect_language(text)
    
    # 3. POS Analysis
    pos = analyze_pos(text)
    
    # 4. NER
    ner = extract_entities(text)
    
    # 5. TF-IDF Keywords
    tfidf = extract_keywords_tfidf(text, top_n=15)
    
    # 6. Coherence & Section Embeddings
    coherence = analyze_section_coherence(text)
    
    # 7. N-Grams
    ngrams = analyze_ngrams(text, max_n=3)
    
    # 8. Language Model / Flow
    lm = compute_language_pattern_score(text)
    
    # 9. Topic Modeling
    topics = discover_topics(text, n_topics=3)
    
    # 10. Sentiment Analysis
    sentiment = analyze_sentiment(text)
    
    # 11. Emotion Detection
    emotion = detect_emotions(text)
    
    # 12. Storytelling (reusing NER and sentiment)
    storytelling = analyze_storytelling(text, ner_results=ner, sentiment_data=sentiment)
    
    # 13. Engagement Analysis
    engagement = analyze_engagement(text)
    
    # 14. Repetition Analysis
    repetition = analyze_repetition(text)
    
    # 15. Complexity Score
    complexity = compute_complexity(text)
    
    # Aggregated dictionary for final scoring
    results_dict = {
        'cleaned_text': cleaned,
        'statistics': stats,
        'language_detection': lang,
        'pos': pos,
        'ner': ner,
        'tfidf': tfidf,
        'coherence': coherence,
        'ngrams': ngrams,
        'language_model': lm,
        'topic_modeling': topics,
        'sentiment': sentiment,
        'emotion': emotion,
        'storytelling': storytelling,
        'engagement': engagement,
        'repetition': repetition,
        'complexity': complexity
    }
    
    # 16. Comprehensive Evaluation Score
    score_evaluation = compute_speech_score(results_dict)
    results_dict['evaluation'] = score_evaluation
    
    return results_dict


# ─── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=64)
    st.title("Speech Setup")
    st.caption("Toastmasters Speech Analysis & Evaluation")
    
    # Dataset loader for preloaded sample speeches
    dataset_df = load_speeches_dataset()
    
    input_mode = st.radio(
        "Choose Input Method",
        ["Select Preloaded Speech", "Paste Transcript", "Upload Text / CSV File"],
        index=0
    )
    
    speech_text = ""
    selected_title = ""
    speaker_name = ""
    
    if input_mode == "Select Preloaded Speech":
        if not dataset_df.empty:
            speech_options = dataset_df['title'].tolist()
            chosen_title = st.selectbox("Select Sample Toastmasters Speech", speech_options)
            row = dataset_df[dataset_df['title'] == chosen_title].iloc[0]
            speech_text = row['text']
            selected_title = row['title']
            speaker_name = row['speaker']
            st.info(f"**Speaker:** {speaker_name}\n\n**Category:** {row['category']}")
        else:
            st.warning("No preloaded speeches found in data/speeches.csv")
            
    elif input_mode == "Paste Transcript":
        selected_title = st.text_input("Speech Title", value="My Toastmasters Speech")
        speaker_name = st.text_input("Speaker Name", value="Toastmaster Speaker")
        speech_text = st.text_area(
            "Paste Speech Transcript Here",
            height=250,
            placeholder="Fellow Toastmasters, honored guests, and friends..."
        )
        
    elif input_mode == "Upload Text / CSV File":
        uploaded_file = st.file_uploader("Upload Speech File", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.txt'):
                speech_text = uploaded_file.read().decode('utf-8', errors='ignore')
                selected_title = uploaded_file.name.replace('.txt', '')
                speaker_name = "Uploaded Speaker"
            elif uploaded_file.name.endswith('.csv'):
                try:
                    up_df = pd.read_csv(uploaded_file)
                    if 'text' in up_df.columns:
                        text_col = 'text'
                        title_col = 'title' if 'title' in up_df.columns else up_df.columns[0]
                        selected_row = st.selectbox("Select Row from CSV", up_df[title_col].tolist())
                        speech_text = up_df[up_df[title_col] == selected_row][text_col].values[0]
                        selected_title = str(selected_row)
                        speaker_name = "CSV Speaker"
                    else:
                        st.error("CSV must contain a 'text' column.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
    
    st.divider()
    st.subheader("⚙️ Coaching Settings")
    
    wpm_rate = st.slider("Target Speaking Rate (WPM)", min_value=110, max_value=180, value=140, step=5,
                         help="Average Toastmasters conversational speaking speed is 130-150 words per minute.")
    
    clf_model_choice = st.selectbox(
        "ML Category Classifier",
        ["logistic_regression", "naive_bayes", "svm"],
        format_func=lambda x: {
            'logistic_regression': 'Logistic Regression (TF-IDF)',
            'naive_bayes': 'Multinomial Naive Bayes',
            'svm': 'Support Vector Machine (Linear SVC)'
        }.get(x, x)
    )
    
    st.markdown("---")
    st.caption("🏫 **PCCOE Pune** | T.Y. B.Tech CSE (AI & ML)\nNLP Mini-Project 2026")


# ─── Main Content Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="academic-pill">🎓 Academic NLP Mini-Project • PCCOE Pune • CSE (AI & ML)</div>
    <h1>ToastMasterAI — Intelligent Speech Coach</h1>
    <p>Comprehensive natural language processing evaluation, linguistic diagnostics, emotional arc tracking, and personalized coaching feedback for Toastmasters speakers.</p>
</div>
""", unsafe_allow_html=True)

# ─── Validation & Trigger ────────────────────────────────────────────────────────
if not speech_text or len(speech_text.strip()) < 40:
    st.info("👋 **Welcome to ToastMasterAI!** Please select a preloaded sample speech from the sidebar or paste your own transcript, then explore the analysis.")
    st.stop()

# Run the NLP Pipeline
with st.spinner("🧠 ToastMasterAI Coach is analyzing speech linguistics, semantics, and delivery patterns..."):
    results = run_nlp_analysis(speech_text, wpm=wpm_rate)
    classifier = get_classifier(clf_model_choice)
    classification_pred = classifier.predict(speech_text)

stats = results['statistics']
eval_score = results['evaluation']
overall_score = eval_score['overall_score']
badge = get_score_badge(overall_score)

# ─── Executive Summary KPI Cards ────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Overall Score</div>
        <div class="metric-value">{overall_score}<span style="font-size:1.1rem; color:#94A3B8;">/100</span></div>
        <div class="metric-sub" style="color:#D97706;">★ {badge}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_words = stats.get('word_count', 0)
    est_sec = int((total_words / wpm_rate) * 60) if wpm_rate > 0 else 0
    est_min_val = round(total_words / wpm_rate, 1) if wpm_rate > 0 else 0
    time_str = stats.get('speaking_time', format_time(est_sec))
    if est_min_val < 4.5:
        bucket = "Short (< 5m)"
    elif est_min_val <= 7.5:
        bucket = "Toastmasters Target (5-7m)"
    else:
        bucket = "Overtime (> 7m)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Speaking Duration</div>
        <div class="metric-value">{time_str}</div>
        <div class="metric-sub">at {wpm_rate} wpm • {bucket}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Word & Sentence Count</div>
        <div class="metric-value">{stats.get('word_count', 0)}</div>
        <div class="metric-sub">{stats.get('sentence_count', 0)} sentences (avg {stats.get('avg_sentence_length', 0):.1f} w/s)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    richness = stats.get('vocabulary_richness', 0.0)
    unique_words_count = stats.get('unique_word_count', 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vocabulary Richness</div>
        <div class="metric-value">{richness:.1f}%</div>
        <div class="metric-sub">{unique_words_count} unique words (TTR)</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    pred_cat = classification_pred['category']
    conf = classification_pred['confidence']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Speech Category</div>
        <div class="metric-value" style="font-size:1.3rem;">{pred_cat}</div>
        <div class="metric-sub">Confidence: {conf}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Navigation Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏆 Scorecard & Coaching",
    "🧠 Linguistic & NLP",
    "🎭 Emotion & Sentiment",
    "📖 Story & Engagement",
    "📊 Vocabulary & Complexity",
    "🤖 ML Classifier",
    "🌐 Web Scraper & Tips",
    "📄 Export Report"
])


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 1: Comprehensive Speech Scorecard & Actionable Coaching
# ═════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🏆 ToastMasterAI Comprehensive Speech Scorecard")
    st.caption("Multidimensional evaluation combining content, organization, rhetoric, vocabulary, and emotional delivery.")
    
    score_cols = st.columns([1.1, 1.3])
    
    category_scores = eval_score['category_scores']
    
    with score_cols[0]:
        # Radar Chart of the 8 Categories
        categories_labels = [
            'Content', 'Organization', 'Vocabulary', 'Language',
            'Engagement', 'Storytelling', 'Emotion', 'Clarity'
        ]
        cat_keys = [
            'content', 'organization', 'vocabulary', 'language',
            'audience_engagement', 'storytelling', 'emotional_impact', 'message_clarity'
        ]
        radar_values = [category_scores.get(k, 0) for k in cat_keys]
        # Close the loop
        radar_values_closed = radar_values + [radar_values[0]]
        radar_labels_closed = categories_labels + [categories_labels[0]]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill='toself',
            fillcolor='rgba(245, 158, 11, 0.25)',
            line=dict(color='#F59E0B', width=2.5),
            name='Speech Score (0-10)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=11, family='Plus Jakarta Sans', color='#1E293B'))
            ),
            showlegend=False,
            margin=dict(l=35, r=35, t=25, b=25),
            height=340
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with score_cols[1]:
        st.write("#### Category Breakdown (0 – 10)")
        c1, c2 = st.columns(2)
        for idx, (cat_name, key) in enumerate(zip(categories_labels, cat_keys)):
            score_val = category_scores.get(key, 0.0)
            target_col = c1 if idx % 2 == 0 else c2
            with target_col:
                st.write(f"**{cat_name}** • `{score_val}/10`")
                st.progress(min(1.0, max(0.0, score_val / 10.0)))
    
    st.divider()
    
    fb_col1, fb_col2 = st.columns(2)
    
    with fb_col1:
        st.write("### 🌟 Key Strengths Identified")
        strengths = eval_score.get('strengths', [])
        if strengths:
            for s in strengths:
                st.markdown(f'<div class="strength-box">✔ {s}</div>', unsafe_allow_html=True)
        else:
            st.info("Your speech scores are evenly balanced across all categories.")
            
        st.write("### 💡 Recommended Areas of Focus")
        improvements = eval_score.get('improvements', [])
        if improvements:
            for imp in improvements:
                st.markdown(f'<div class="improve-box">▲ {imp}</div>', unsafe_allow_html=True)
        else:
            st.success("Congratulations! All core categories are performing strongly above threshold.")

    with fb_col2:
        st.write("### 🎯 Actionable Coaching Feedback")
        feedback_list = eval_score.get('feedback', [])
        if feedback_list:
            for idx, fb in enumerate(feedback_list):
                st.markdown(f"""
                <div class="feedback-box">
                    <strong>Coach Tip #{idx+1}:</strong> {fb}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("Keep practicing consistent vocal variety and structured pauses for optimal speech delivery.")
            
    st.markdown("""
    <div class="disclaimer-banner">
        ⚖️ <strong>Academic Disclaimer:</strong> ToastMasterAI is an educational natural language processing tool developed for the T.Y. B.Tech CSE (AI & ML) NLP curriculum at PCCOE Pune. It does not replace human Toastmasters International evaluators or official contest judging.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 2: Linguistic & NLP Deep Dive
# ═════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🧠 NLP Deep Dive & Grammatical Diagnostics")
    st.caption("Granular natural language processing results: POS tagging, NER, TF-IDF, N-Grams, and semantic coherence.")
    
    nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs([
        "🏷️ POS & Style", "🏷️ Named Entities (NER)", "🔑 TF-IDF & Topics", "🔗 Coherence & Embeddings"
    ])
    
    with nlp_tab1:
        pos_data = results['pos']
        st.write("#### Part-of-Speech Distribution & Speaking Style")
        
        pos_cols = st.columns([1.2, 1])
        with pos_cols[0]:
            pos_dist = pos_data.get('percentages', {})
            df_pos = pd.DataFrame({
                'Part of Speech': list(pos_dist.keys()),
                'Percentage (%)': list(pos_dist.values())
            }).sort_values(by='Percentage (%)', ascending=False)
            
            fig_pos = px.bar(
                df_pos, x='Part of Speech', y='Percentage (%)',
                color='Percentage (%)',
                color_continuous_scale='Blues',
                text='Percentage (%)',
                title="POS Grammatical Profile"
            )
            fig_pos.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_pos.update_layout(height=320, showlegend=False, margin=dict(t=35, b=20))
            st.plotly_chart(fig_pos, use_container_width=True)
            
        with pos_cols[1]:
            st.write("##### Speaking Style Interpretation")
            interpretations = pos_data.get('interpretations', [])
            for interp in interpretations:
                st.info(f"📌 {interp}")
                
            st.write(f"• **Adjective/Adverb Ratio:** {pos_data.get('adjective_adverb_ratio', 0):.1f}%")
            st.write(f"• **Dynamic Verb Frequency:** {pos_data.get('verb_action_frequency', 0):.1f}%")

    with nlp_tab2:
        ner_data = results['ner']
        st.write("#### Named Entities Discovered in Speech")
        st.caption("Identified people, organizations, geopolitical entities, and temporal milestones.")
        
        n_c1, n_c2, n_c3, n_c4 = st.columns(4)
        n_c1.metric("People Mentioned", len(ner_data.get('people', [])))
        n_c2.metric("Organizations", len(ner_data.get('organizations', [])))
        n_c3.metric("Locations", len(ner_data.get('locations', [])))
        n_c4.metric("Dates / Time Refs", len(ner_data.get('dates', [])))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        ent_by_type = ner_data.get('entity_by_type', {})
        if ent_by_type:
            for etype, items in ent_by_type.items():
                if items:
                    st.write(f"**{etype} ({len(items)}):**")
                    chips = "".join([f'<span class="badge-chip" style="background:#E2E8F0; color:#1E293B;">{it}</span>' for it in items[:15]])
                    st.markdown(chips, unsafe_allow_html=True)
                    st.write("")
        else:
            st.info("No distinct named entities detected. Consider adding personal names or specific place settings to ground your story.")

    with nlp_tab3:
        tfidf_data = results['tfidf']
        topic_data = results['topic_modeling']
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.write("#### Top TF-IDF Keywords")
            keywords = tfidf_data.get('keywords', [])
            if keywords:
                df_kw = pd.DataFrame(keywords[:10])
                fig_kw = px.bar(
                    df_kw, x='score', y='word', orientation='h',
                    color='score', color_continuous_scale='Oranges',
                    title="TF-IDF Salience Scores"
                )
                fig_kw.update_layout(yaxis=dict(autorange="reversed"), height=320, showlegend=False)
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No significant keywords extracted.")
                
        with t_col2:
            st.write("#### Discovered Latent Topics (LDA)")
            st.write(f"**Primary Topic:** `{topic_data.get('primary_topic', 'General Speech')}`")
            
            topics_list = topic_data.get('topics', [])
            if topics_list:
                for t in topics_list:
                    with st.expander(f"📌 {t.get('label', 'Topic')} ({t.get('weight', 0):.1f}%)"):
                        st.write(f"**Salient Words:** {', '.join(t.get('words', []))}")
            else:
                st.info("Not enough text variation to extract multiple topics.")

    with nlp_tab4:
        coherence_data = results['coherence']
        st.write("#### Semantic Coherence & Speech Architecture")
        
        coh_c1, coh_c2 = st.columns(2)
        with coh_c1:
            st.metric("Opening ↔ Conclusion Similarity", f"{coherence_data.get('opening_conclusion_similarity', 0)}%")
            st.caption("High circularity indicates that the conclusion reinforces the opening hook.")
            
            st.metric("Section-to-Section Flow", f"{coherence_data.get('avg_coherence', 0)}%")
            st.caption("Average cosine similarity between consecutive speech sections.")
            
        with coh_c2:
            sec_sims = coherence_data.get('section_similarities', {})
            if sec_sims:
                df_sims = pd.DataFrame({
                    'Transition': list(sec_sims.keys()),
                    'Continuity (%)': list(sec_sims.values())
                })
                fig_coh = px.bar(
                    df_sims, x='Transition', y='Continuity (%)',
                    color='Continuity (%)', color_continuous_scale='Teal',
                    title="Thematic Transition Continuity"
                )
                fig_coh.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_coh, use_container_width=True)
                
        interps = coherence_data.get('interpretation', [])
        for inp in interps:
            st.info(f"💡 {inp}")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 3: Emotion & Sentiment
# ═════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎭 Emotional Arc & Sentiment Analysis")
    st.caption("Sentence-by-sentence sentiment dynamics and NRC emotion tracking.")
    
    sent_data = results['sentiment']
    emo_data = results['emotion']
    
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Overall Sentiment", sent_data.get('overall_sentiment', 'Neutral'))
    s_col2.metric("Dominant Emotion", emo_data.get('dominant_emotion', 'Neutral').title())
    s_col3.metric("Positivity Compound Score", f"{sent_data.get('compound_score', 0):.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Emotional Arc Timeline Plot
    timeline = sent_data.get('timeline', [])
    if timeline:
        df_timeline = pd.DataFrame({
            'Sentence Number': list(range(1, len(timeline) + 1)),
            'Sentiment Polarity': timeline
        })
        fig_arc = px.line(
            df_timeline, x='Sentence Number', y='Sentiment Polarity',
            title="Speech Emotional Journey (Sentence-by-Sentence Sentiment Arc)",
            markers=True
        )
        fig_arc.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Neutral Line")
        fig_arc.update_traces(line_color="#2563EB", line_width=2.5)
        fig_arc.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_arc, use_container_width=True)
        
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.write("#### Emotion Breakdown")
        emo_dist = emo_data.get('distribution', {})
        if emo_dist:
            df_emo = pd.DataFrame({
                'Emotion': [k.title() for k in emo_dist.keys()],
                'Intensity (%)': list(emo_dist.values())
            }).sort_values(by='Intensity (%)', ascending=False)
            
            fig_emo = px.bar(
                df_emo, x='Emotion', y='Intensity (%)',
                color='Emotion',
                title="Relative Emotional Resonance"
            )
            fig_emo.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_emo, use_container_width=True)
            
    with e_col2:
        st.write("#### Emotional Arc Interpretation")
        emo_interps = emo_data.get('interpretations', [])
        for e_inp in emo_interps:
            st.markdown(f'<div class="feedback-box">🎭 {e_inp}</div>', unsafe_allow_html=True)
            
        sent_dist = sent_data.get('distribution', {})
        st.write(f"• **Positive Sentences:** {sent_dist.get('positive', 0)}%")
        st.write(f"• **Neutral Sentences:** {sent_dist.get('neutral', 0)}%")
        st.write(f"• **Negative Sentences:** {sent_dist.get('negative', 0)}%")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 4: Story & Engagement
# ═════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📖 Storytelling & Audience Engagement")
    st.caption("Toastmasters storytelling elements (Hook, Struggle, Turning Point, Call-to-Action).")
    
    story_data = results['storytelling']
    eng_data = results['engagement']
    
    st_c1, st_c2 = st.columns(2)
    with st_c1:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:1rem;">
            <div class="metric-label">Storytelling Score</div>
            <div class="metric-value">{story_data.get('story_score', 0)}<span style="font-size:1.1rem; color:#94A3B8;">/100</span></div>
            <div class="metric-sub">{len(story_data.get('story_sentences', []))} narrative sentences detected</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("##### Story Elements Detected")
        st.write(f"• **Characters / Names:** {len(story_data.get('characters', []))}")
        st.write(f"• **Location Settings:** {len(story_data.get('locations', []))}")
        st.write(f"• **Conflict / Challenge Mentions:** {len(story_data.get('problems', []))}")
        st.write(f"• **Resolution Statements:** {len(story_data.get('resolutions', []))}")
        
        st.write("##### Narrative Coaching Feedback")
        for s_inp in story_data.get('interpretations', []):
            st.info(f"📖 {s_inp}")
            
    with st_c2:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:1rem;">
            <div class="metric-label">Audience Engagement Score</div>
            <div class="metric-value">{eng_data.get('score', 0)}<span style="font-size:1.1rem; color:#94A3B8;">/100</span></div>
            <div class="metric-sub">{eng_data.get('questions', 0)} audience questions • {eng_data.get('audience_references', 0)} direct references</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("##### Engagement Drivers")
        st.write(f"• **Rhetorical / Audience Questions:** {eng_data.get('questions', 0)}")
        st.write(f"• **Inclusive Pronouns ('you', 'we'):** {eng_data.get('audience_references', 0)}")
        st.write(f"• **Calls to Action (CTA):** {eng_data.get('calls_to_action', 0)}")
        
        st.write("##### Engagement Suggestions")
        for e_sug in eng_data.get('suggestions', []):
            st.warning(f"🎯 {e_sug}")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 5: Vocabulary, Complexity & Repetition
# ═════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📊 Vocabulary, Complexity & Ah-Counter Diagnostics")
    st.caption("Toastmasters Ah-Counter simulation, word variety, and readability metrics.")
    
    comp_data = results['complexity']
    rep_data = results['repetition']
    
    v_c1, v_c2, v_c3 = st.columns(3)
    v_c1.metric("Speech Complexity", f"{comp_data.get('overall_score', 0)}/100", comp_data.get('label', 'Balanced'))
    v_c2.metric("Repetition Index", f"{rep_data.get('repetition_score', 0)}/100")
    
    fillers = stats.get('filler_words', {})
    total_fillers = sum(fillers.values())
    v_c3.metric("Filler Words (Ah-Counter)", total_fillers, "Flagged" if total_fillers > 3 else "Clear")
    
    st.divider()
    
    vc_col1, vc_col2 = st.columns(2)
    with vc_col1:
        st.write("#### 🛑 Ah-Counter: Filler Words Identified")
        if fillers:
            df_fillers = pd.DataFrame({
                'Filler Word': list(fillers.keys()),
                'Count': list(fillers.values())
            }).sort_values(by='Count', ascending=False)
            
            fig_fil = px.bar(
                df_fillers, x='Filler Word', y='Count',
                color='Count', color_continuous_scale='Reds',
                title="Filler Word Frequency"
            )
            fig_fil.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_fil, use_container_width=True)
        else:
            st.success("🎉 Excellent! Zero filler words detected in the transcript.")
            
        overused = stats.get('overused_words', {})
        if overused:
            st.write("##### Frequently Repeated Common Words")
            st.write(", ".join([f"`{w}` ({c}x)" for w, c in overused.items()]))

    with vc_col2:
        st.write("#### 🔁 Repeated Phrases & Starters")
        rep_phrases = rep_data.get('repeated_phrases', [])
        if rep_phrases:
            for rp in rep_phrases[:5]:
                st.write(f"• **\"{rp.get('phrase')}\"** — repeated {rp.get('count')} times")
        else:
            st.write("No excessive multi-word phrase repetition detected.")
            
        st.write("##### Complexity Component Breakdown")
        comp_comps = comp_data.get('components', {})
        for cname, cval in comp_comps.items():
            st.write(f"• {cname.replace('_', ' ').title()}: `{cval:.1f}`")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 6: ML Classifier & Category Insights
# ═════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 Machine Learning Speech Category Classifier")
    st.caption("Supervised classification across 9 Toastmasters pathways/speech categories.")
    
    clf_c1, clf_c2 = st.columns([1, 1.2])
    
    with clf_c1:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:1.2rem;">
            <div class="metric-label">Predicted Category</div>
            <div class="metric-value">{classification_pred['category']}</div>
            <div class="metric-sub">Model: {clf_model_choice.replace('_', ' ').title()} • Confidence: {classification_pred['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("##### Active Model Architecture")
        st.write("• **Feature Extraction:** TF-IDF Vectorizer (ngram_range=(1,2), max_features=1000)")
        st.write(f"• **Classifier Algorithm:** {classification_pred.get('model_type', 'Logistic Regression').title()}")
        st.write(f"• **Cross-Validation Accuracy:** {classification_pred.get('model_accuracy', 45.0)}%")

    with clf_c2:
        st.write("#### Category Probability Distribution")
        probs = classification_pred.get('all_probabilities', {})
        if probs:
            df_probs = pd.DataFrame({
                'Category': list(probs.keys()),
                'Probability (%)': list(probs.values())
            }).sort_values(by='Probability (%)', ascending=True)
            
            fig_prob = px.bar(
                df_probs, x='Probability (%)', y='Category', orientation='h',
                color='Probability (%)', color_continuous_scale='Viridis',
                title="Prediction Confidence per Category"
            )
            fig_prob.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Probability distribution unavailable for this model.")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 7: Web Scraper & Toastmasters Resources
# ═════════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🌐 Public Toastmasters Resource Explorer & Scraper")
    st.caption("Academic web scraping tool for permitted, publicly available public speaking resources and tips.")
    
    scrape_url = st.text_input(
        "Enter Permitted Public Webpage URL (e.g. public speech tips or article)",
        value="https://en.wikipedia.org/wiki/Toastmasters_International"
    )
    
    if st.button("🌐 Scrape & Analyze Webpage Content"):
        with st.spinner("Fetching and processing public webpage content..."):
            scrape_res = scrape_page(scrape_url)
            if scrape_res.get('success'):
                st.success(f"Successfully scraped: **{scrape_res.get('title')}** ({scrape_res.get('word_count')} words)")
                with st.expander("📄 View Scraped Text Content"):
                    st.write(scrape_res.get('text')[:2000] + "...")
            else:
                st.error(f"Scraping failed: {scrape_res.get('error')}")


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 8: Export Evaluation Report
# ═════════════════════════════════════════════════════════════════════════════════
with tab8:
    st.subheader("📄 Export Comprehensive Speech Evaluation Report")
    st.caption("Generate a downloadable markdown/text evaluation report for the speaker's portfolio.")
    
    report_text = f"""# ToastMasterAI — Speech Evaluation Report
**Speech Title:** {selected_title}
**Speaker:** {speaker_name}
**Date of Evaluation:** 2026-10-10
**Evaluated by:** ToastMasterAI NLP Speech Coach (PCCOE Pune)

---

## 1. Executive Summary
- **Overall Evaluation Score:** {overall_score} / 100 ({badge})
- **Speaking Time:** {stats.get('speaking_time', 'N/A')} (at {wpm_rate} WPM)
- **Word Count:** {stats['word_count']} words across {stats['sentence_count']} sentences
- **Vocabulary Richness (TTR):** {stats['vocabulary_richness']:.1f}%
- **Predicted Category:** {classification_pred['category']} ({classification_pred['confidence']}% confidence)

---

## 2. Category Scores (0 - 10)
- **Content:** {category_scores.get('content', 0)} / 10
- **Organization:** {category_scores.get('organization', 0)} / 10
- **Vocabulary:** {category_scores.get('vocabulary', 0)} / 10
- **Language Quality:** {category_scores.get('language', 0)} / 10
- **Audience Engagement:** {category_scores.get('audience_engagement', 0)} / 10
- **Storytelling:** {category_scores.get('storytelling', 0)} / 10
- **Emotional Impact:** {category_scores.get('emotional_impact', 0)} / 10
- **Message Clarity:** {category_scores.get('message_clarity', 0)} / 10

---

## 3. Key Strengths
{chr(10).join(['- ' + s for s in eval_score.get('strengths', ['Solid overall presentation.'])])}

---

## 4. Areas for Improvement
{chr(10).join(['- ' + imp for imp in eval_score.get('improvements', ['Continue refining delivery style.'])])}

---

## 5. Personalized Coaching Recommendations
{chr(10).join([f'{i+1}. ' + fb for i, fb in enumerate(eval_score.get('feedback', ['Practice deliberate pauses.']))])}

---

## 6. Linguistic & Delivery Diagnostics
- **Dominant Emotion:** {emo_data.get('dominant_emotion', 'Neutral').title()}
- **Overall Sentiment:** {sent_data.get('overall_sentiment', 'Neutral')}
- **Filler Words (Ah-Counter):** {total_fillers} flagged
- **Opening ↔ Conclusion Circularity:** {results['coherence'].get('opening_conclusion_similarity', 0)}%

---
*Notice: This evaluation was generated using natural language processing models for academic demonstration at PCCOE Pune.*
"""

    st.download_button(
        label="📥 Download Full Evaluation Report (.md)",
        data=report_text,
        file_name=f"ToastMasterAI_Report_{selected_title.replace(' ', '_')}.md",
        mime="text/markdown"
    )
    
    with st.expander("👁️ Preview Full Report"):
        st.markdown(report_text)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.85rem; padding: 1rem 0;">
    <strong>ToastMasterAI</strong> — Academic NLP Mini-Project | <strong>T.Y. B.Tech CSE (AI & ML)</strong> | <strong>PCCOE Pune</strong><br>
    Built with Python, NLTK, Scikit-Learn, Streamlit, and Plotly.
</div>
""", unsafe_allow_html=True)
