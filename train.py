"""
Training Script for ToastMasterAI
Trains the speech classifier on the sample dataset.

Usage:
    python train.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_speeches_dataset
from classification.speech_classifier import SpeechClassifier


def main():
    print("=" * 60)
    print("ToastMasterAI — Model Training")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading speech dataset...")
    df = load_speeches_dataset()
    
    if df.empty:
        print("ERROR: No speech dataset found at data/speeches.csv")
        print("Please ensure the dataset file exists.")
        return
    
    print(f"  Loaded {len(df)} speeches")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  Category distribution:")
    for cat, count in df['category'].value_counts().items():
        print(f"    - {cat}: {count}")
    
    texts = df['text'].tolist()
    labels = df['category'].tolist()
    
    # Train Logistic Regression
    print("\n[2/4] Training Logistic Regression classifier...")
    lr_classifier = SpeechClassifier(model_type='logistic_regression')
    lr_result = lr_classifier.train(texts, labels)
    print(f"  Training complete. Accuracy: {lr_result.get('accuracy', 'N/A')}%")
    
    # Save model
    model_path = lr_classifier.save_model('speech_classifier.pkl')
    print(f"  Model saved to: {model_path}")
    
    # Train Naive Bayes
    print("\n[3/4] Training Naive Bayes classifier...")
    nb_classifier = SpeechClassifier(model_type='naive_bayes')
    nb_result = nb_classifier.train(texts, labels)
    print(f"  Training complete. Accuracy: {nb_result.get('accuracy', 'N/A')}%")
    nb_classifier.save_model('speech_classifier_nb.pkl')
    
    # Train SVM
    print("\n[4/4] Training SVM classifier...")
    svm_classifier = SpeechClassifier(model_type='svm')
    svm_result = svm_classifier.train(texts, labels)
    print(f"  Training complete. Accuracy: {svm_result.get('accuracy', 'N/A')}%")
    svm_classifier.save_model('speech_classifier_svm.pkl')
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    test_text = texts[0] if texts else "This is a test speech about leadership and growth."
    
    for name, clf in [('Logistic Regression', lr_classifier), 
                       ('Naive Bayes', nb_classifier),
                       ('SVM', svm_classifier)]:
        result = clf.predict(test_text)
        print(f"\n  {name}:")
        print(f"    Predicted: {result['category']}")
        print(f"    Confidence: {result['confidence']}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNote: This sample dataset is for academic demonstration purposes.")
    print("The classifier accuracy may improve with more training data.")


if __name__ == '__main__':
    main()
