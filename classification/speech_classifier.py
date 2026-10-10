"""
Speech Classifier Module for ToastMasterAI
Trains and uses ML models to classify speeches into categories.

Categories: Personal Development, Leadership, Communication, Persuasive,
            Educational, Humor, Experience Sharing, Teamwork, Motivation
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from utils.helpers import SPEECH_CATEGORIES


# Path for saved models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


class SpeechClassifier:
    """
    Speech type classifier using TF-IDF features and ML models.
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Args:
            model_type: 'logistic_regression', 'naive_bayes', or 'svm'
        """
        self.model_type = model_type
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.accuracy = 0.0
        self.categories = SPEECH_CATEGORIES
    
    def _create_pipeline(self):
        """Create the sklearn pipeline based on model type."""
        tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        
        if self.model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=1.0)
        elif self.model_type == 'svm':
            classifier = LinearSVC(max_iter=1000, random_state=42)
        else:
            classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', classifier)
        ])
    
    def train(self, texts: list, labels: list) -> dict:
        """
        Train the classifier on speech texts and labels.
        
        Args:
            texts: List of speech text strings
            labels: List of category labels
        
        Returns:
            Dictionary with training results
        """
        if not texts or not labels or len(texts) != len(labels):
            return {'success': False, 'error': 'Invalid training data'}
        
        self._create_pipeline()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Train
        self.pipeline.fit(texts, encoded_labels)
        self.is_trained = True
        
        # Cross-validation (if enough samples)
        if len(texts) >= 5:
            cv_folds = min(5, len(texts))
            scores = cross_val_score(self.pipeline, texts, encoded_labels, cv=cv_folds)
            self.accuracy = round(float(scores.mean()) * 100, 1)
        else:
            self.accuracy = 0.0
        
        return {
            'success': True,
            'model_type': self.model_type,
            'num_samples': len(texts),
            'num_categories': len(set(labels)),
            'categories': list(set(labels)),
            'accuracy': self.accuracy
        }
    
    def predict(self, text: str) -> dict:
        """
        Predict the category of a speech.
        
        Args:
            text: Speech text
        
        Returns:
            Dictionary with predicted category and confidence
        """
        if not self.is_trained or self.pipeline is None:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {},
                'message': 'Model not trained. Please run training first.'
            }
        
        try:
            # Predict
            prediction = self.pipeline.predict([text])[0]
            category = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get probabilities (if available)
            probabilities = {}
            if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
                probs = self.pipeline.predict_proba([text])[0]
                classes = self.label_encoder.inverse_transform(self.pipeline.named_steps['classifier'].classes_)
                probabilities = {
                    cls: round(float(prob) * 100, 1)
                    for cls, prob in sorted(zip(classes, probs), key=lambda x: -x[1])
                }
                confidence = round(float(max(probs)) * 100, 1)
            elif hasattr(self.pipeline.named_steps['classifier'], 'decision_function'):
                # SVM uses decision function
                decision = self.pipeline.decision_function([text])[0]
                if isinstance(decision, np.ndarray):
                    # Multi-class SVM
                    exp_scores = np.exp(decision - np.max(decision))
                    pseudo_probs = exp_scores / exp_scores.sum()
                    classes = self.label_encoder.inverse_transform(
                        self.pipeline.named_steps['classifier'].classes_
                    )
                    probabilities = {
                        cls: round(float(prob) * 100, 1)
                        for cls, prob in sorted(zip(classes, pseudo_probs), key=lambda x: -x[1])
                    }
                    confidence = round(float(max(pseudo_probs)) * 100, 1)
                else:
                    confidence = round(min(100, abs(float(decision)) * 20), 1)
            else:
                confidence = self.accuracy  # Fallback
            
            return {
                'category': category,
                'confidence': confidence,
                'all_probabilities': probabilities,
                'model_type': self.model_type,
                'model_accuracy': self.accuracy
            }
        
        except Exception as e:
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {},
                'message': f'Prediction error: {str(e)}'
            }
    
    def save_model(self, filename: str = 'speech_classifier.pkl'):
        """Save trained model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        filepath = os.path.join(MODEL_DIR, filename)
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'is_trained': self.is_trained,
            'categories': self.categories
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return filepath
    
    def load_model(self, filename: str = 'speech_classifier.pkl') -> bool:
        """Load a trained model from disk."""
        filepath = os.path.join(MODEL_DIR, filename)
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.accuracy = model_data['accuracy']
            self.is_trained = model_data['is_trained']
            self.categories = model_data.get('categories', SPEECH_CATEGORIES)
            return True
        except Exception:
            return False
