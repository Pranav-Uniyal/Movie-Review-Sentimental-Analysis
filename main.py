import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Configuration (Matches your .h5 file's 5000-word shape)
VOCAB_SIZE = 5000  
MAX_LEN = 200
EMBEDDING_DIM = 128

@st.cache_resource
def load_resources():
    # Load Model Weights
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.4),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    try:
        model.load_weights("sentiment_analysis_model.h5")
    except Exception as e:
        st.error(f"Model Error: {e}")

    # Load Tokenizer
    try:
        # Use the NEWly saved tokenizer file name here
        with open("tokenizer_new.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"Tokenizer Error: {e}")
        tokenizer = None
        
    return model, tokenizer

# Initialize
model, tokenizer = load_resources()

# UI
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter review:", height=150)

if st.button("Analyze"):
    if review.strip() and tokenizer:
        with st.spinner("Analyzing..."):
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = model.predict(padded)[0][0]
            
            res = "Positive" if pred > 0.5 else "Negative"
            color = "green" if res == "Positive" else "red"
            st.markdown(f"### Sentiment: :{color}[{res}]")
            st.write(f"Confidence Score: {pred if res == 'Positive' else 1-pred:.2f}")
    else:
        st.warning("Please enter text and ensure tokenizer is uploaded.")
