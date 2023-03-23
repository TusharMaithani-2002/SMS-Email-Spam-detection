import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")


def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        # removing special characters
        # checkigng for alpha numeric columns
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


if st.button('Predict'):
    # preprocess
    transformed_sms = tranform_text(str(input_sms))
    # vectorize 
    vector_input = tfidf.transform([transformed_sms])
    # print(len(vector_input))
    # predict
    result = model.predict(vector_input)
    print(result)
    # display

    if result[0] == 1:
        st.header('spam')
    else:
        st.header('not spam')

