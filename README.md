# ToastMasterAI: An Intelligent NLP-Based Toastmasters Speech Analysis and Coaching System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20Scikit--Learn-green.svg)](https://www.nltk.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Academic Project](https://img.shields.io/badge/PCCOE-Pune-orange.svg)](https://www.pccoepune.com/)

> **Academic Mini-Project**  
> **Course:** Natural Language Processing (NLP)  
> **Department:** Computer Engineering (Artificial Intelligence & Machine Learning)  
> **Class:** T.Y. B.Tech CSE (AI & ML)  
 

---

## 📖 1. Project Overview

**ToastMasterAI** is an intelligent, end-to-end natural language processing coaching system tailored specifically for Toastmasters public speakers. Rather than presenting natural language processing techniques as disconnected classroom demonstrations, ToastMasterAI deploys a suite of **18 integrated NLP modules working behind the scenes** to evaluate speech transcripts and provide actionable, objective coaching feedback.

### Key Capabilities
- 🎙️ **Speech Duration & Pacing Estimation:** Computes total speaking time and Toastmasters timer intervals (Green, Amber, Red flag zones) based on customizable speaking rates (120–180 WPM).
- 🏆 **Comprehensive 8-Dimension Evaluation Scorecard:** Computes category scores (Content, Organization, Vocabulary, Language, Audience Engagement, Storytelling, Emotional Impact, and Message Clarity) culminating in a weighted overall 0–100 score.
- 🎭 **Emotional Arc & Sentiment Journey:** Tracks sentence-by-sentence polarity progression and multi-class NRC emotion distribution (Joy, Trust, Anticipation, Fear, Sadness, Anger).
- 🏷️ **Grammatical & Syntactic Profiling:** Extracts POS ratios (dynamic verbs vs. descriptive adjectives), identifies Named Entities (NER), and uncovers latent themes using LDA Topic Modeling.
- 🔗 **Semantic Coherence & Structure:** Measures thematic circularity between the opening hook and conclusion, as well as adjacent section transitions.
- 🛑 **Ah-Counter & Repetition Diagnostics:** Identifies filler words (*um, uh, like, actually, you know*), repeated n-gram phrases, and sentence starters.
- 🤖 **Pathways ML Classifier:** Supervised speech classification across 9 Toastmasters pathways/speech categories using Logistic Regression, Naive Bayes, and Linear Support Vector Machines (SVM).
- 📄 **Exportable Reports:** Generates structured, downloadable evaluation reports in Markdown format.

---

## 🏛️ 2. Architectural Blueprint

```
Speech-Coach-/
├── app.py                      # Main Streamlit web application & interactive dashboard
├── train.py                    # Model training pipeline for speech classification
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation
├── data/
│   └── speeches.csv            # Labeled Toastmasters speech dataset (16 speeches, 9 categories)
├── preprocessing/
│   ├── cleaner.py              # Text normalization, punctuation & contraction handling
│   ├── tokenizer.py            # Sentence & word tokenization
│   ├── stemmer.py              # PorterStemmer module
│   └── lemmatizer.py           # WordNet lemmatizer & POS-aware alternatives
├── analysis/
│   ├── statistics.py           # Word count, speaking duration, TTR lexical richness
│   ├── language_detection.py   # Statistical & heuristic language identifier
│   ├── pos_analysis.py         # POS tag distribution & speaking style interpretation
│   ├── ner.py                  # Named Entity Recognition (People, Orgs, Locations, Dates)
│   ├── tfidf.py                # TF-IDF keyword extraction & salience scoring
│   ├── embeddings.py           # Vector embeddings & semantic coherence
│   ├── ngrams.py               # Bigram/Trigram phrase pattern extraction
│   ├── language_model.py       # N-gram language model & predictability scoring
│   ├── topic_modeling.py       # Latent Dirichlet Allocation (LDA) topic discovery
│   ├── sentiment.py            # VADER sentence-level & speech-level sentiment
│   ├── emotion.py              # Multi-class emotion detection & narrative arc
│   ├── storytelling.py         # Narrative hooks, conflict, and resolution detection
│   ├── engagement.py           # Rhetorical questions, inclusive pronouns, and CTAs
│   ├── repetition.py           # Repeated n-grams and repetitive sentence starters
│   └── complexity.py           # Readability & linguistic complexity index
├── classification/
│   └── speech_classifier.py    # Sklearn ML pipeline (TF-IDF + LR / NB / SVM)
├── evaluation/
│   └── speech_score.py         # Multi-factor weighted rubric & feedback generator
├── scraping/
│   └── scraper.py              # Polite web scraper for public Toastmasters resources
├── utils/
│   └── helpers.py              # Shared scoring weights, formatting & dataset utilities
└── models/
    ├── speech_classifier.pkl   # Trained Logistic Regression model
    ├── speech_classifier_nb.pkl# Trained Naive Bayes model
    └── speech_classifier_svm.pkl# Trained Linear SVM model
```

---

## 🔬 3. NLP Modules Explained

| # | Module | NLP Technique / Model | Educational Purpose |
|---|---|---|---|
| **1** | `preprocessing/cleaner.py` | Regex normalization, lowercase conversion, whitespace stripping | Standardizes raw speech transcripts for downstream processing. |
| **2** | `preprocessing/tokenizer.py` | NLTK `sent_tokenize`, `word_tokenize`, regex word boundary matching | Deconstructs continuous speech into discrete sentence & word tokens. |
| **3** | `preprocessing/stemmer.py` | Porter Stemmer Algorithm | Demonstrates morphological suffix stripping. |
| **4** | `preprocessing/lemmatizer.py` | NLTK WordNet Lemmatizer with Treebank-to-WordNet POS mapping | Preserves semantic dictionary lemmas and retrieves synonyms. |
| **5** | `analysis/statistics.py` | Type-Token Ratio (TTR), WPM duration calculation | Measures lexical variety and Toastmasters timing constraints. |
| **6** | `analysis/language_detection.py` | `langdetect` + Stopword heuristic fallback | Confirms transcript language and flags non-English input. |
| **7** | `analysis/pos_analysis.py` | NLTK `pos_tag` (Penn Treebank tagset) | Evaluates rhetorical style (e.g. action verbs vs. descriptive adjectives). |
| **8** | `analysis/ner.py` | Named Entity Recognition (spaCy / NLTK `ne_chunk`) | Identifies characters, organizations, locations, and temporal markers. |
| **9** | `analysis/tfidf.py` | `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) | Extracts the most salient keywords and core themes. |
| **10** | `analysis/embeddings.py` | TF-IDF Subspace Vectorization + Cosine Similarity | Analyzes opening-to-conclusion circularity and flow across sections. |
| **11** | `analysis/ngrams.py` | Unigram, Bigram, Trigram extraction with frequency counting | Uncovers common phrase patterns, rhetoric, and rhythmic cadence. |
| **12** | `analysis/language_model.py` | N-gram Language Modeling with Laplace (Add-1) Smoothing | Computes transition predictability and speech fluency. |
| **13** | `analysis/topic_modeling.py` | Latent Dirichlet Allocation (LDA) via CountVectorizer | Discovers hidden thematic topics and semantic cluster distributions. |
| **14** | `analysis/sentiment.py` | NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) | Generates compound sentiment scores and sentence-by-sentence arcs. |
| **15** | `analysis/emotion.py` | NRC-style Multi-Class Emotion Lexicon matching | Maps emotions (Joy, Trust, Fear, Sadness, Anger) across the speech. |
| **16** | `analysis/storytelling.py` | Rule-based regex pattern matching + NER + Sentiment correlation | Detects storytelling elements: hooks, struggles, turning points, resolutions. |
| **17** | `analysis/engagement.py` | Question detection, 2nd-person pronoun analysis, Call to Action flags | Measures audience involvement, rhetorical queries, and audience connection. |
| **18** | `analysis/repetition.py` | N-gram repetition index & sentence starter frequency analysis | Detects rhetorical anaphora vs. unintentional verbal crutches. |
| **19** | `analysis/complexity.py` | Composite NLP Readability Index (Sentence length, rare words, TTR) | Assesses whether language complexity matches spoken delivery standards. |
| **20** | `classification/speech_classifier.py`| Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machine | Classifies speech transcripts into 9 distinct Toastmasters categories. |
| **21** | `evaluation/speech_score.py` | Multi-factor weighted rubric & rule-based coaching expert system | Computes 0–100 overall score and generates 3–5 personalized recommendations. |
| **22** | `scraping/scraper.py` | `requests` + `BeautifulSoup4` with polite delay & User-Agent | Collects publicly permitted public speaking articles and sample texts. |

---

## ⚖️ 4. Evaluation Rubric & Scoring Weights

The final comprehensive score ($0 - 100$) is computed as a weighted combination of 8 individual category evaluations ($0 - 10$):

$$\text{Overall Score} = \sum_{i=1}^{8} \left( \text{Score}_i \times \text{Weight}_i \times 10 \right)$$

| Dimension | Default Weight | Key NLP Signals |
|---|:---:|---|
| **Content** | 20% | Word count suitability, keyword salience, vocabulary richness |
| **Organization** | 15% | Opening-conclusion thematic similarity, transition coherence |
| **Vocabulary** | 15% | Type-Token Ratio (TTR), absence of filler words, overuse penalties |
| **Language Quality** | 10% | Balanced sentence complexity, language model predictability |
| **Audience Engagement** | 10% | Question density, inclusive pronouns ("you", "we"), call-to-action |
| **Storytelling** | 10% | Narrative character count, conflict markers, resolution statements |
| **Emotional Impact** | 10% | Emotion variety, sentiment arc trajectory (uplifting resolution) |
| **Message Clarity** | 10% | TF-IDF concentration, LDA primary topic prominence |

---

## 🚀 5. Installation & Execution Guide

### Prerequisites
- Python 3.8 to 3.13
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/Akshit926/Speech-Coach-.git
cd Speech-Coach-
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Train Speech Classifiers (Optional if pre-trained)
```bash
python train.py
```
*This trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM models on `data/speeches.csv` and serializes them to the `models/` directory.*

### Step 4: Launch the Streamlit Coaching Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 6. Sample Dataset

The included `data/speeches.csv` dataset contains 16 curated, full-length Toastmasters speeches across 9 speech pathways:
1. **Personal Development** (*e.g., "The Power of Vulnerability", "Embracing the Uncomfortable"*)
2. **Leadership** (*e.g., "The Quiet Leader", "Leading Through Uncertainty"*)
3. **Communication** (*e.g., "The Art of Listening", "Words That Heal"*)
4. **Persuasive** (*e.g., "The Case for Curiosity", "Dare to Say No"*)
5. **Educational** (*e.g., "How Habits Shape Our Destiny", "The Psychology of First Impressions"*)
6. **Humor** (*e.g., "My Adventures in Public Speaking", "The Day My GPS Betrayed Me"*)
7. **Experience Sharing** (*e.g., "Lessons from a Marathon Runner"*)
8. **Teamwork** (*e.g., "The Symphony of Collaboration"*)
9. **Motivation** (*e.g., "The Spark Within"*)

---

## 🎓 7. Academic Integrity & Disclaimer

- **Academic Context:** Developed by T.Y. B.Tech Computer Engineering (AI & ML) students at Pimpri Chinchwad College of Engineering (PCCOE), Pune for the Natural Language Processing laboratory coursework.
- **Disclaimer:** ToastMasterAI is an educational research and development prototype. The scoring models and metrics are generated algorithmically using computational linguistics and do not represent official Toastmasters International judging criteria or contest scorecards.
- **Toastmasters Recognition:** Toastmasters International and Toastmasters Pathways are registered trademarks of Toastmasters International.

---

## 👥 Contributors
- **Akshit & Team** — T.Y. B.Tech CSE (AI & ML), PCCOE Pune
- **Course Instructor / Evaluator:** Department of Computer Engineering, PCCOE Pune
