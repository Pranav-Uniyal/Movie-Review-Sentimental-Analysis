import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import sys

# --- FORCED PATCH FOR KEYERROR/MODULE NOT FOUND ---
# This creates a fake 'keras.src.legacy' path so the old pickle can load
import keras
if not hasattr(keras, "src"):
    # If using an older Keras version where .src doesn't exist
    pass
else:
    # Create the path that the old tokenizer is looking for
    sys.modules['keras.src'] = keras.src
    sys.modules['keras.src.legacy'] = keras.src

# Configuration
VOCAB_SIZE = 5000  
MAX_LEN = 200
EMBEDDING_DIM = 128

@st.cache_resource
def load_resources():
    # 1. Rebuild Architecture
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.4),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # 2. Load Weights (Fixed name)
    model_file = "sentiment_analysis_model.h5"
    if os.path.exists(model_file):
        model.load_weights(model_file)
    else:
        st.error(f"Missing file: {model_file}")

    # 3. Load Tokenizer (Back to your original file name)
    tokenizer_file = "tokenizer.pkl"
    tokenizer = None
    if os.path.exists(tokenizer_file):
        try:
            with open(tokenizer_file, "rb") as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            st.error(f"Tokenizer Load Error: {e}")
    else:
        st.error(f"Missing file: {tokenizer_file}")
        
    return model, tokenizer

# Initialize
model, tokenizer = load_resources()

# UI
st.title("🎬 Movie Review Sentiment Analysis")
review = st.text_area("Enter a movie review:", height=150)

if st.button("Analyze"):
    if review.strip() and tokenizer:
        with st.spinner("Analyzing..."):
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            pred = model.predict(padded)[0][0]
            
            res = "Positive" if pred > 0.5 else "Negative"
            color = "green" if res == "Positive" else "red"
            st.markdown(f"### Sentiment: :{color}[{res}]")
            st.write(f"**Confidence:** {pred if res == 'Positive' else 1-pred:.2f}")
    elif not tokenizer:
        st.error("Tokenizer not loaded. Check your GitHub files.")
    else:
        st.warning("Please enter some text.")

st.divider()
st.caption("Built with Streamlit & TensorFlow")
