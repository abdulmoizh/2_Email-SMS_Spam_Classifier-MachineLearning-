# app.py
import os
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')


# --- NLTK data directory inside the repo (so it can be committed) ---
HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
NLTK_DATA_DIR = os.path.join(HERE, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Ensure NLTK searches our local nltk_data first
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Download resources only if missing (quietly)
def ensure_nltk(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name.split('/')[-1], download_dir=NLTK_DATA_DIR, quiet=True)

# Required resources
ensure_nltk("tokenizers/punkt")      # punkt tokenizer (word_tokenize / sent_tokenize)
ensure_nltk("corpora/stopwords")    # stopwords
ensure_nltk("corpora/wordnet")      # optional - you had it in original code

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    # keep only alphanumeric tokens
    for token in tokens:
        if token.isalnum():
            y.append(token)

    text = y[:]   # copy
    y.clear()

    # remove stopwords and punctuation
    stops = set(stopwords.words('english'))
    for token in text:
        if token not in stops and token not in string.punctuation:
            y.append(token)

    text = y[:]
    y.clear()

    # stemming
    for token in text:
        y.append(ps.stem(token))

    return " ".join(y)

# --- load pickles (fail fast with helpful message) ---
VEC_PATH = os.path.join(HERE, "vectorizer.pkl")
MODEL_PATH = os.path.join(HERE, "model.pkl")

if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
    missing = []
    if not os.path.exists(VEC_PATH):
        missing.append("vectorizer.pkl")
    if not os.path.exists(MODEL_PATH):
        missing.append("model.pkl")
    st.error(f"Missing files in repo: {', '.join(missing)}. Put them in the app folder and redeploy.")
    st.stop()

with open(VEC_PATH, 'rb') as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# --- Streamlit UI ---
st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms or input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # preprocess
        transform_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transform_sms])

        # predict
        result = model.predict(vector_input)[0]

        # display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

