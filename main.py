import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import sys

# --- UPDATED CONFIGURATION ---
# Fixed to 5000 based on your error message
VOCAB_SIZE = 5000  
MAX_LEN = 200
EMBEDDING_DIM = 128

# Fix for "No module named 'keras.src.legacy'"
# We tell Python to treat 'keras.src' as a valid path if it's missing
if 'keras' in sys.modules:
    import keras
    if not hasattr(keras, 'src'):
        sys.modules['keras.src.legacy'] = keras

@st.cache_resource
def get_model():
    model_path = "sentiment_analysis_model.h5"
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.4),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    try:
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None

@st.cache_resource
def get_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Tokenizer Error: {e}. If this persists, you may need to re-save your tokenizer using the new Keras version.")
        return None

# Load Resources
model = get_model()
tokenizer = get_tokenizer()

# --- UI LOGIC ---
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter your review:", height=150)

if st.button("Analyze"):
    if review.strip() and model and tokenizer:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        prediction = model.predict(padded)[0][0]
        
        label = "Positive" if prediction > 0.5 else "Negative"
        st.markdown(f"### Sentiment: **{label}**")
        st.write(f"Confidence: {prediction if label == 'Positive' else 1-prediction:.2f}")
    else:
        st.warning("Ensure text is entered and model/tokenizer are loaded.")
