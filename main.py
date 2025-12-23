import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import urllib.request

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_analysis_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
# Replace the URL below with your actual GitHub Raw link if deployment fails
RAW_MODEL_URL = "https://github.com/Pranav-Uniyal/Movie-Review-Sentimental-Analysis/raw/main/sentiment_analysis_model.h5"

@st.cache_resource
def get_model():
    # Fix for Git LFS: Check if file is just a pointer (tiny size)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        with st.spinner("Downloading full model file..."):
            urllib.request.urlretrieve(RAW_MODEL_URL, MODEL_PATH)
    
    # safe_mode=False is required for Keras 3 to load legacy .h5 files
    return load_model(MODEL_PATH, compile=False, safe_mode=False)

@st.cache_resource
def get_tokenizer():
    with open(TOKENIZER_PATH, "rb") as handle:
        return pickle.load(handle)

# Initialization
model = get_model()
tokenizer = get_tokenizer()

# UI
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter review:", height=150)

def predict_sentiment(text):
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=200)
        prediction = model.predict(padded)
        
        score = prediction[0][0]
        sentiment = "Positive" if score > 0.5 else "Negative"
        confidence = score if score > 0.5 else 1 - score
        return sentiment, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

if st.button("Analyze"):
    if review.strip():
        with st.spinner("Processing..."):
            res, conf = predict_sentiment(review)
            if res:
                color = "green" if res == "Positive" else "red"
                st.markdown(f"### Sentiment: :{color}[{res}]")
                st.write(f"**Confidence:** {conf:.2f}")
    else:
        st.warning("Please enter text.")

st.divider()
st.caption("Built with Streamlit & TensorFlow")
