import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    # Load and preprocess
    df = pd.read_csv("imdb_dataset.csv")
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
    plt.show()

    return nb_model, tfidf

def predict_sentiment(review, model, tfidf):
    review_tfidf = tfidf.transform([review])
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

if __name__ == "__main__":
    model, tfidf_vectorizer = train_and_evaluate()
    
    # Test prediction
    sample_text = "The movie was boring and a complete waste of time"
    result = predict_sentiment(sample_text, model, tfidf_vectorizer)
    print(f"\nReview: {sample_text}")
    print(f"Sentiment: {result}")
