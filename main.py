import pandas as pd
import numpy as np
import streamlit as st
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


# intializing some fucntions for text preprocessing
stop_words = set(stopwords.words("english"))
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Load the model
model = load_model('LSTM_emotion_model.h5')

#load the encoder
with open('label_encoder.pkl', 'rb') as handle:
    le = pickle.load(handle)

# multiple funciton to clean the text data
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
            
def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

# main prediction function
def analysis(sentence):
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    print(result, proba)
    return result, proba

st.title('Sentiment Analysis From Text')
st.caption('This is a simple sentiment analysis web app that uses a LSTM model to predict the sentiment of a given text')

sentence = st.text_input('Enter your text here', key='text')
emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
emotions_text = '\n\n'.join(emotions).upper()
with st.sidebar:
    st.title("About")
    st.link_button("GitHub", "https://github.com/Shaheer04", use_container_width=True)
    st.link_button("LinkedIn", "https://www.linkedin.com/in/shaheerjamal/", use_container_width=True)
    st.markdown("## Emotions")
    st.info(emotions_text)
    st.caption("Made with ❤️ by Shaheer Jamal") 


if st.button('Analyze Text', type="primary"):
    with st.spinner('Analyzing the text'):
        result, proba = analysis(sentence)
        st.success(result.upper())

        progress_text = 'Prediction.'
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(int(proba*100)):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

    st.success(f'The probability of the sentiment is: {int(proba*100)} %')
    st.caption('Made with ❤️ by Shaheer Jamal')