import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load Model
model_path = r'C:\Users\loku0\OneDrive\Desktop\Movie Sentimental\sentiment_analysis_model.h5' # you can add path of model according to your directory
                                                                                              # where you have saved it  
try:
    model = load_model(model_path)
except OSError:
    st.error("Error loading the model. Please check if the file exists and is not corrupted.")

# Load Tokenizer
tokenizer_path = r'C:\Users\loku0\OneDrive\Desktop\Movie Sentimental\tokenizer.pkl'
try:
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Tokenizer file not found. Please check the path.")

# App Title
st.title("Movie Review Sentiment Analysis")

# App Description
st.write("""
### Analyze if a movie review is positive or negative""")

# Input Box
review = st.text_area("Enter a movie review:")

# Prediction Function
def predict_sentiment(text):
    try:
        # Preprocess the input
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=200)  # Assuming maxlen used in training was 200

        # Predict sentiment
        prediction = model.predict(padded)
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
        confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Predict Button
if st.button("Analyze"):
    if review.strip() == "":
        st.error("Please enter a valid review!")
    else:
        sentiment, confidence = predict_sentiment(review)
        if sentiment:
            st.write(f"### Sentiment: {sentiment}")
            st.write(f"### Confidence: {confidence:.2f}")

# Footer
st.write("This is the sentimental review.")
