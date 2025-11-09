import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punk_tab')


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:] # : COLON IS INMPORTANT TO PUT HERE BECAUSE STRING IS A MUTABLE DATATYPE AND WHENEVER YOU'LL THE "TEXT" BE COPYING IT 
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # step 1. preprocess
    transform_sms = transform_text(input_sms)

    # step 2. vectorize
    vector_input = tfidf.transform([transform_sms])

    # step 3. predict
    result = model.predict(vector_input)

    # step 1. display
    if result == 1:
        st.header("Spam")
    else:

        st.header("Not Spam")

