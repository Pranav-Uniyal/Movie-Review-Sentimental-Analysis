import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# --- MODEL CONFIGURATION ---
# These MUST match the settings used during training
VOCAB_SIZE = 5000 
MAX_LEN = 200
EMBEDDING_DIM = 128

@st.cache_resource
def get_model():
    model_path = "sentiment_analysis_model.h5"
    
    # 1. Manually Rebuild the Architecture
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.4),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # 2. Load Weights Only (Bypasses the TypeError)
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
        st.error(f"Tokenizer error: {e}")
        return None

# Load resources
model = get_model()
tokenizer = get_tokenizer()

# --- APP UI ---
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter your movie review:", height=150)

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter some text.")
    elif model and tokenizer:
        with st.spinner("Analyzing..."):
            # Preprocessing
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            
            # Prediction
            prediction = model.predict(padded)[0][0]
            
            # Results display
            res = "Positive" if prediction > 0.5 else "Negative"
            color = "green" if res == "Positive" else "red"
            
            st.markdown(f"### Sentiment: :{color}[{res}]")
            st.progress(float(prediction) if res == "Positive" else 1.0 - float(prediction))
            st.write(f"**Confidence Score:** {prediction:.2f}")

