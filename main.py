import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# Ensure nltk punkt tokenizer is available
nltk.download('punkt')

# Streamlit App Configuration
st.title("😊 Text Sentiment Analysis App")
st.write("Analyze the sentiment of your text input (positive, neutral, or negative) and visualize the results!")

# User Input
user_input = st.text_area("Enter your text below:", "")

# Add a Submit Button
if st.button("Submit"):
    if user_input.strip():
        # Perform Sentiment Analysis
        blob = TextBlob(user_input)
        sentences = blob.sentences
        polarities = [sentence.sentiment.polarity for sentence in sentences]
        
        # Display Sentiment Result
        st.subheader("Sentiment Analysis Result")
        if polarities:
            avg_sentiment = sum(polarities) / len(polarities)
            sentiment_label = "Positive 😊" if avg_sentiment > 0 else "Negative 😞" if avg_sentiment < 0 else "Neutral 😐"
            st.write(f"Overall Sentiment: **{sentiment_label}**")
            st.write(f"Average Polarity Score: {avg_sentiment:.2f}")
        else:
            st.write("No sentences detected for sentiment analysis.")

        # Bar Chart for Polarity Scores
        st.subheader("📊 Polarity Score Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, len(polarities) + 1), polarities, color='cornflowerblue')
        ax.set_title("Polarity Score for Each Sentence")
        ax.set_xlabel("Sentence Number")
        ax.set_ylabel("Polarity Score")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)  # Add a horizontal line at 0
        st.pyplot(fig)

        # Word Cloud Visualization
        st.subheader("📊 Word Cloud Visualization")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.error("Please enter some text before submitting.")
