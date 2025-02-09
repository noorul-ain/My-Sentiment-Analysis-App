import streamlit as st
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download("punkt")

# Streamlit app setup
st.title("Sentiment Analysis")
st.write("Generate a word cloud to visualize the sentiment of words in your input text.")

# User input
user_input = st.text_area("Enter your text:", placeholder="Type your text here...")

# Function for word-level sentiment analysis
def analyze_words(text):
    tokens = nltk.word_tokenize(text)
    word_sentiments = {"Positive": [], "Negative": [], "Neutral": []}
    
    for word in tokens:
        blob = TextBlob(word)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            word_sentiments["Positive"].append(word)
        elif polarity < 0:
            word_sentiments["Negative"].append(word)
        else:
            word_sentiments["Neutral"].append(word)
    
    return word_sentiments

# Analyze button
if st.button("Analyze"):
    if user_input.strip():
        word_sentiments = analyze_words(user_input)

        # Display the results
        st.subheader("Word Sentiment Analysis")
        for sentiment, words in word_sentiments.items():
            st.write(f"**{sentiment} Words:** {' '.join(words) if words else 'None'}")
        
        # Generate word clouds for each sentiment
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        colormaps = {"Positive": "Greens", "Negative": "Reds", "Neutral": "gray"}

        for i, (sentiment, words) in enumerate(word_sentiments.items()):
            if words:  # Only generate a word cloud if there are words for the sentiment
                wordcloud = WordCloud(width=400, height=400, background_color="white", colormap=colormaps[sentiment]).generate(" ".join(words))
                axs[i].imshow(wordcloud, interpolation="bilinear")
                axs[i].set_title(f"{sentiment} Words", fontsize=16)
                axs[i].axis("off")
            else:
                axs[i].set_title(f"No {sentiment} Words", fontsize=16)
                axs[i].axis("off")

        st.pyplot(fig)
    else:
        st.warning("Please enter some text to analyze.")
