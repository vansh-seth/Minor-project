import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import webbrowser
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# API Key
API_KEY = "AIzaSyDwmbrqHT490eJhcyolc-8nKqrpW5J_324"
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

# Function to fetch fact-check results
@st.cache_data
def check_news_fact(query):
    params = {"query": query, "key": API_KEY, "languageCode": "en"}
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        results = []
        all_claims_text = ""

        if "claims" in data:
            for claim in data["claims"]:
                claim_text = claim["text"]
                all_claims_text += claim_text + " "
                sentiment = get_sentiment(claim_text)

                for review in claim.get("claimReview", []):
                    results.append([
                        claim_text,
                        review["publisher"]["name"],
                        review["textualRating"],
                        review["url"],
                        sentiment
                    ])

            df = pd.DataFrame(results, columns=["Claim", "Source", "Rating", "URL", "Sentiment"])
            return df, all_claims_text
    return None, None

# Streamlit App
st.title("🕵️ Fact-Check Analyzer")
st.markdown("🔎 **Enter a news claim to check its credibility!**")

query = st.text_input("Enter a claim to check:", "COVID-19 vaccines cause infertility")

if st.button("Check Fact"):
    df, all_claims_text = check_news_fact(query)

    if df is not None:
        # Show Data Table
        st.subheader("🔍 Fact-Check Results")
        st.dataframe(df)

        # Save CSV
        csv_file = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Download Results as CSV", data=csv_file, file_name=f"fact_check_{query.replace(' ', '_')}.csv", mime="text/csv")

        # Sentiment Analysis Pie Chart
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig = px.pie(values=sentiment_counts, names=sentiment_counts.index, title="Sentiment Analysis of Claims", color=sentiment_counts.index, color_discrete_map={"Positive": "green", "Neutral": "blue", "Negative": "red"})
        st.plotly_chart(fig)

        # Fact Check Ratings Bar Chart
        st.subheader("📊 Fact-Check Ratings")
        rating_counts = df["Rating"].value_counts()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"},
            title="Distribution of Fact-Check Ratings",
            color=rating_counts.index
            )
        
        max_length = 10
        shortened_labels = [label[:max_length] for label in rating_counts.index]
        fig.update_layout(
            xaxis=dict(
                tickvals=rating_counts.index,  # Set positions of ticks
                ticktext=shortened_labels,      # Set shortened labels
                tickfont=dict(size=10)          # Optional: Adjust font size as needed
                )
            )
        st.plotly_chart(fig)


        # Word Cloud of Claims
        st.subheader("☁️ Word Cloud of Checked Claims")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_claims_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    else:
        st.error("❌ No fact-check results found for this claim.")
