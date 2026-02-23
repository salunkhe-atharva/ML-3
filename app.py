import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- UI Header ---
st.title("🎬 IMDB Sentiment Analyzer")
st.write("Loading model, please wait...")

@st.cache_resource # This stops the app from retraining every time you type
def train_and_evaluate():
    current_dir = Path(__file__).parent
    data_path = current_dir / "imdb_dataset.csv"

    if not data_path.exists():
        st.error(f"File not found! Make sure 'imdb_dataset.csv' is in your GitHub repo.")
        st.stop()

    df = pd.read_csv(data_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    return nb_model, tfidf

# Load data/model
model, tfidf_vectorizer = train_and_evaluate()
st.success("Model ready!")

# --- User Interaction Section ---
user_input = st.text_area("Enter a movie review:", "This movie was fantastic!")

if st.button("Predict Sentiment"):
    review_tfidf = tfidf_vectorizer.transform([user_input])
    prediction = model.predict(review_tfidf)
    result = "Positive" if prediction[0] == 1 else "Negative"
    
    color = "green" if result == "Positive" else "red"
    st.markdown(f"### Sentiment: :{color}[{result}]")
