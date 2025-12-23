import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sentiment_analysis_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Use st.cache_resource to load the model only once and improve performance
@st.cache_resource
def load_my_model(path):
    try:
        # 'compile=False' avoids errors related to training-specific settings 
        # that aren't needed for prediction.
        # 'safe_mode=False' allows loading legacy architecture configurations.
        return load_model(path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_resource
def load_tokenizer(path):
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Tokenizer file error: {e}")
        return None

# Perform the actual loading
model = load_my_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)

# App Title
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Analyze if a movie review is positive or negative")

# Input Box
review = st.text_area("Enter a movie review:", height=150)

# Prediction Function
def predict_sentiment(text):
    if not model or not tokenizer:
        st.error("Model or Tokenizer not properly loaded.")
        return None, None
        
    try:
        # Preprocess the input
        sequences = tokenizer.texts_to_sequences([text])
        # Assuming maxlen used in training was 200
        padded = pad_sequences(sequences, maxlen=200)

        # Predict sentiment
        prediction = model.predict(padded)
        
        # Standard binary classification threshold
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Predict Button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence = predict_sentiment(review)
            
            if sentiment:
                color = "green" if sentiment == "Positive" else "red"
                st.markdown(f"### Sentiment: :{color}[{sentiment}]")
                st.progress(float(confidence))
                st.write(f"**Confidence Level:** {confidence:.2f}")

# Footer
st.divider()
st.caption("Built with Streamlit & TensorFlow")
