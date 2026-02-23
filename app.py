import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    # 1. Define the correct path
    current_dir = Path(__file__).parent
    data_path = current_dir / "imdb_dataset.csv"

    # 2. Safety Check: If file is missing, list all files to help you debug
    if not data_path.exists():
        files_found = os.listdir(current_dir)
        raise FileNotFoundError(f"Could not find {data_path.name}. Files in directory: {files_found}")

    # 3. Load and preprocess
    df = pd.read_csv(data_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X = df['review']
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # Predict
    y_pred = nb_model.predict(X_test_tfidf)
    
    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Plotting
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.close() # Close plot to save memory on server

    return nb_model, tfidf

# Keep your predict_sentiment and __main__ blocks as they were
