import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Movie Sentiment Analyzer 🎬",
    page_icon="🎥",
    layout="centered"
)

# =====================================================
# Custom CSS (UI MAGIC ✨)
# =====================================================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3, h4, p, label {
    color: #e5e7eb !important;
}
.card {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(255, 215, 0, 0.15);
    margin-top: 20px;
}
.rating {
    font-size: 40px;
    font-weight: bold;
    color: gold;
}
.sentiment-positive {
    color: #22c55e;
    font-size: 26px;
    font-weight: bold;
}
.sentiment-negative {
    color: #ef4444;
    font-size: 26px;
    font-weight: bold;
}
.footer {
    color: #94a3b8;
    text-align: center;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Load model & tokenizer (Cloud Safe)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sentiment_analysis_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

@st.cache_resource
def load_resources():
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# =====================================================
# Header
# =====================================================
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Analyze movie reviews with **AI-powered sentiment detection** & **IMDB-style ratings** ⭐")

# =====================================================
# Input
# =====================================================
review = st.text_area(
    "✍️ Enter a movie review",
    height=160,
    placeholder="This movie was absolutely amazing with stunning performances..."
)

# =====================================================
# Prediction Logic
# =====================================================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    prob = model.predict(padded, verbose=0)[0][0]

    sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
    rating = round(prob * 10, 1) if prob > 0.5 else round((1 - prob) * 10, 1)

    return sentiment, prob, rating

# =====================================================
# Button
# =====================================================
if st.button("🎯 Analyze Review"):
    if not review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        sentiment, confidence, rating = predict_sentiment(review)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Sentiment
        if "Positive" in sentiment:
            st.markdown(f'<div class="sentiment-positive">🎉 {sentiment}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sentiment-negative">💔 {sentiment}</div>', unsafe_allow_html=True)

        # Rating
        st.markdown(f'<div class="rating">⭐ {rating} / 10</div>', unsafe_allow_html=True)

        # Confidence bar
        st.progress(float(confidence if confidence > 0.5 else 1 - confidence))

        st.write(f"**Model Confidence:** `{max(confidence, 1-confidence):.2f}`")

        # Verdict
        if rating >= 8:
            st.success("🔥 Blockbuster – Highly Recommended!")
        elif rating >= 6:
            st.info("👍 Worth Watching")
        else:
            st.warning("👎 Not Recommended")

        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# Footer
# =====================================================
st.markdown('<div class="footer">Built with ❤️ using TensorFlow & Streamlit</div>', unsafe_allow_html=True)
