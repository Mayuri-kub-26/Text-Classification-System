import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_data
from preprocess import preprocess_text

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_classifier.pkl')

def train_and_evaluate():
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    if df is None:
        print("Could not load data. Exiting.")
        return

    # 2. Preprocess Data
    print("Preprocessing data (this might take a moment)...")
    # Apply preprocessing to a new column
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    X = df['processed_message']
    y = df['label']

    # 3. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build Pipeline
    # Using TF-IDF and Naive Bayes as it's standard for text classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # 5. Train Model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 7. Visualization (Confusion Matrix)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

    # 8. Save Model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
