# 🎬 Movie Review Sentimental Analysis | [Live](https://movie-review-sentimental-analysis-hkbwzmp9jcacgk5dyo3hle.streamlit.app/)

This project is a Streamlit-based web application that performs **sentiment analysis** on user-written movie reviews. It uses a pre-trained deep learning model (Keras with TensorFlow backend) to classify reviews as either **Positive** or **Negative**, and displays a corresponding **confidence score**.

---

## 🧠 Overview

Using techniques from natural language processing (NLP) and deep learning, this app turns raw text into model-ready input and interprets sentiment in real-time. It's designed to demonstrate how deep learning can help understand human opinions in the context of movie reviews.

---

## 💡 Features

- 📝 Enter any custom movie review
- 🤖 Pretrained LSTM-based sentiment classifier
- 📊 Displays classification result with confidence
- 🧼 Tokenization and padding using saved tokenizer
- 🚀 Clean, interactive UI built with Streamlit

---

## 📁 Project Structure

```
Movie-Review-Sentimental-Analysis/
├── main.py                         # Streamlit app script
├── sentiment_analysis_model.h5     # Pre-trained sentiment model
├── tokenizer.pkl                   # Tokenizer used during training
├── README.md                       # Project documentation (this file)                                 
└── Movie_Sentiment_NB              #Jupyter Notebook where model is created and trained 
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pranav-Uniyal/Movie-Review-Sentimental-Analysis.git
cd Movie-Review-Sentimental-Analysis
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit tensorflow pickle5
```

---

## ▶️ Running the App

To launch the web app locally:

```bash
streamlit run main.py
```

Once launched, your browser will automatically open the app at `http://localhost:8501`.

---

## 🧠 How It Works

1. The app loads a pre-trained Keras model and tokenizer (`tokenizer.pkl`).
2. When the user inputs a review, it is tokenized and padded to match the input format of the model (`maxlen=200`).
3. The model predicts the sentiment:
   - **Positive** if prediction > 0.5
   - **Negative** otherwise
4. The result and confidence score are displayed in real-time.

---

## 📝 Example

**Input:**

> “The cinematography was breathtaking, but the story lacked depth.”

**Output:**
>![image](https://github.com/user-attachments/assets/edb181c3-6381-4526-b8da-fbf63ec9437d)

 ---
**Similarly Other Examples**
>![image](https://github.com/user-attachments/assets/8f1e1f9e-8b26-4bf7-866c-73d8b5f7d314)
---
>![image](https://github.com/user-attachments/assets/61c74c20-594e-47a0-85f0-b2db68c71538)

---

## ⚠️ Common Issues

- **Model Not Loading:** Ensure the `sentiment_analysis_model.h5` path is correct.
- **Tokenizer Not Found:** Make sure `tokenizer.pkl` exists and is properly referenced.
- **Mismatch in `maxlen`:** Ensure the same padding length (`maxlen=200`) used during training is used during inference.

---

## 🚀 Future Enhancements

- Add a third "Neutral" class
- Batch prediction from CSV file
- Add visualizations (word clouds, attention maps)
- Improve UI design with themes and transitions

---

## 👨‍💻 Developed By

**Pranav Uniyal**  
GitHub: [Pranav-Uniyal](https://github.com/Pranav-Uniyal)

---

## 📄 License

This project is intended for educational purposes.  
Feel free to fork, modify, and use with attribution.
