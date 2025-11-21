# app.py - Streamlit UI for your saved spam model
import streamlit as st
import pickle
import numpy as np
import re
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec

# --- ensure NLTK stopwords are available (first run may download) ---
nltk.download('stopwords', quiet=True)

# ---------- LOAD MODELS ----------
BASE_DIR = os.path.dirname(__file__)  # folder where app.py sits

# Try to load classifier - change name if yours is different
clf_path = os.path.join(BASE_DIR, "spam_model.pkl")
if not os.path.exists(clf_path):
    raise FileNotFoundError(f"Classifier file not found: {clf_path}")
classifier = pickle.load(open(clf_path, "rb"))

# Load Word2Vec model
w2v_path = os.path.join(BASE_DIR, "word2vec.model")
if not os.path.exists(w2v_path):
    raise FileNotFoundError(f"Word2Vec model file not found: {w2v_path}")
w2v_model = Word2Vec.load(w2v_path)

# Try to load stopwords and stemmer pickles if they exist, else create fresh
stopwords_path = os.path.join(BASE_DIR, "stopwords.pkl")
stemmer_path = os.path.join(BASE_DIR, "stemmer.pkl")

if os.path.exists(stopwords_path):
    stop_words = pickle.load(open(stopwords_path, "rb"))
else:
    stop_words = set(stopwords.words('english'))

if os.path.exists(stemmer_path):
    ps = pickle.load(open(stemmer_path, "rb"))
else:
    ps = PorterStemmer()

# ---------- PREPROCESSING ----------
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    tokens = text.lower().split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return tokens

def avg_word2vec(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv.key_to_index]
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 Spam Email / SMS Detector")

st.write("Paste an email or SMS below and click Predict. Model: Word2Vec + RandomForest.")

user_input = st.text_area("Enter message here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please type a message first.")
    else:
        tokens = preprocess_text(user_input)
        vec = avg_word2vec(tokens).reshape(1, -1)
        pred = classifier.predict(vec)[0]  # 0 = ham, 1 = spam
        proba = None
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(vec)[0].max()

        if pred == 1:
            st.error("⚠️ This looks like **SPAM**!")
        else:
            st.success("✅ This looks **NOT SPAM (HAM)**.")

        if proba is not None:
            st.write(f"Model confidence: {proba*100:.2f}%")
