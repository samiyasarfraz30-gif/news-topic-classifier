# ==========================================
# IMPORT LIBRARIES
# ==========================================

from flask import Flask, render_template, request

import pickle
import re
import string
import os

import nltk

# ==========================================
# DOWNLOAD NLTK DATA (ONCE, WITH CORRECT PATH)
# ==========================================

# Set NLTK data path explicitly for Render
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_PATH)

# Download required NLTK packages
for package in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(package, download_dir=NLTK_DATA_PATH, quiet=True)
    except Exception as e:
        print(f"Warning: Could not download {package}: {e}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ==========================================
# LOAD MODEL & VECTORIZER (ABSOLUTE PATHS)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    vectorizer = None

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# NLP TOOLS
# ==========================================

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ==========================================
# CATEGORY LABELS
# ==========================================

category_map = {
    1: "🌍 World",
    2: "⚽ Sports",
    3: "💹 Business",
    4: "💻 Sci/Tech"
}

# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    cleaned_words = []
    for word in words:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            cleaned_words.append(stemmed_word)

    return " ".join(cleaned_words)

# ==========================================
# PREDICTION HISTORY
# ==========================================

history = []

# ==========================================
# HOME ROUTE
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    confidence = 0
    user_text = ""
    probabilities = {}
    word_count = 0
    char_count = 0
    token_count = 0
    result_color = "#22c55e"
    error_message = ""

    if request.method == 'POST':
        try:
            user_text = request.form['news_text']

            if model is None or vectorizer is None:
                error_message = "Model or vectorizer failed to load. Please check server logs."
            else:
                # Statistics
                char_count = len(user_text)
                word_count = len(user_text.split())

                # Preprocess
                cleaned_text = preprocess_text(user_text)
                token_count = len(cleaned_text.split())

                # TF-IDF
                vector_input = vectorizer.transform([cleaned_text])

                # Prediction
                result = model.predict(vector_input)[0]
                prediction = category_map[result]

                # Probabilities
                probs = model.predict_proba(vector_input)[0]
                confidence = round(max(probs) * 100, 2)

                probabilities = {
                    "🌍 World":    round(probs[0] * 100, 2),
                    "⚽ Sports":   round(probs[1] * 100, 2),
                    "💹 Business": round(probs[2] * 100, 2),
                    "💻 Sci/Tech": round(probs[3] * 100, 2)
                }

                # Dynamic colors
                if "World" in prediction:
                    result_color = "#f59e0b"
                elif "Sports" in prediction:
                    result_color = "#10b981"
                elif "Business" in prediction:
                    result_color = "#3b82f6"
                elif "Sci/Tech" in prediction:
                    result_color = "#a855f7"

                # Save history
                history.append({
                    "text": user_text[:60] + "...",
                    "prediction": prediction
                })

        except Exception as e:
            error_message = f"Prediction error: {str(e)}"
            print(f"❌ Prediction error: {e}")

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        user_text=user_text,
        probabilities=probabilities,
        word_count=word_count,
        char_count=char_count,
        token_count=token_count,
        history=history[-5:],
        result_color=result_color,
        error_message=error_message
    )

# ==========================================
# RUN APP
# ==========================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)