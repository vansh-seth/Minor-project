import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK Setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define cleaning function for text data
def clean_text(text):
    text = text.lower()
    # Expand contractions
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    # Remove special characters and extra spaces
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize, remove stopwords, and lemmatize
    cleaned_words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(cleaned_words)

# Load Data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake['category'] = 1  # Fake news
true['category'] = 0  # True news
df = pd.concat([fake, true]).reset_index(drop=True)

# Clean the data
df['text'] = df['text'].apply(clean_text)

# Model Building
X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Build the model pipeline
text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
text_clf.fit(X_train, y_train)

# Calibrate model
calibrated_clf = CalibratedClassifierCV(text_clf.named_steps["clf"], cv="prefit")
calibrated_clf.fit(text_clf.named_steps["tfidf"].transform(X_train), y_train)

# Define function to predict fake news
def predict_fake_news(news):
    prediction = text_clf.predict([news])
    return "Fake" if prediction == 1 else "True"

# Streamlit UI
st.title("Fake News Detection")
st.markdown("## Check if a news article is fake or not by entering the text below")

# User Input
user_input = st.text_area("Enter News Article Here", height=200)

if st.button("Check News"):
    if user_input:
        # Predict the news
        prediction = predict_fake_news(user_input)

        st.subheader(f"The news is: **{prediction}**")

        # Sentiment Analysis using TextBlob
        sentiment_score = TextBlob(user_input).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        st.write(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")

        # Word Cloud of User Input
        st.subheader("Word Cloud of the News Text")
        wordcloud = WordCloud(max_words=100, width=800, height=400).generate(user_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Visualizing the Sentiment Distribution for Fake vs True News
        sentiment_distribution = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['sentiment'] = sentiment_distribution
        plt.figure(figsize=(10,5))
        sns.histplot(data=df, x="sentiment", hue="category", bins=50, kde=True, palette=["red", "blue"])
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.title("Sentiment Distribution in Fake vs True News")
        st.pyplot(plt)

        # Accuracy Metrics for Model
        st.subheader("Model Performance Metrics")
        predictions = text_clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(metrics.classification_report(y_test, predictions))
        st.write(metrics.confusion_matrix(y_test, predictions))

    else:
        st.error("Please enter some text to analyze.")
