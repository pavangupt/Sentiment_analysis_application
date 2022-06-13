import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download(["stopwords", "wordnet"])
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
wordnet = WordNetLemmatizer()


def text_preprocessing(data):
    sent = data["Text"]
    corpus = []
    for i in range(len(sent)):
        words = re.sub("^a-zA-Z", " ", str(sent[i]))
        words = words.lower()
        words = words.split()
        words = [wordnet.lemmatize(word) for word in words if not word in set(stopwords.words("english"))]
        words = " ".join(words)
        corpus.append(words)
    return corpus


def sentiment_analyser(data):
    sent_list = []
    for sent in data["Text"]:
        sentiment_score = TextBlob(str(sent)).sentiment[0]  # to get the sentiment of sentence
        if sentiment_score > 0:
            sent_list.append("Positive Review")
        elif sentiment_score < 0:
            sent_list.append("Negative Review")
        else:
            sent_list.append("Neural Review")
    return sent_list


def data_sentiment(df):
    df = df.dropna()
    df.reset_index(inplace=True)
    corpus_list = text_preprocessing(df)
    corpus = pd.DataFrame(corpus_list, columns=["Text"])
    df_txt = pd.concat([df["ID"], corpus, df["Star"]], axis=1)
    sentiment_list = sentiment_analyser(df_txt)  # This function will give us the sentiment of each sentence
    s = pd.DataFrame(sentiment_list, columns=["Sentiment"])
    final_data = pd.concat([df_txt, s], axis=1)
    final_data = final_data[final_data["Sentiment"] == "Positive Review"]
    final_data = final_data[final_data["Star"] == 1]
    return final_data


st.title("Sentiment Analyzer Application")
st.sidebar.title("Sentiment Analysis")
st.sidebar.image("download.jpg")
st.write("Important :- Please upload a file whose column name are [Text] and [Star]!")
uploaded_file = st.file_uploader("Choose a csv file!")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df, 2000, 200)
    df_final = data_sentiment(df)
    if st.button("Check Sentiment"):
        st.write(" Below dataframe represent ratings where review text is good, but rating is very low!")
        st.dataframe(df_final, 2000, 400)
