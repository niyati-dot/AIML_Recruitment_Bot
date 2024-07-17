import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
import requests
import re
import urllib.parse as p
import csv
from urllib.error import HTTPError
import glob
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import digits
from cleantext import clean
stop_words = set(stopwords.words('english'))

df = pd.read_csv("./train_sentiment.csv",encoding= 'unicode_escape')
text_df = df[['text','sentiment']]
print(text_df.shape)
text_df = text_df[text_df['sentiment'] != 'neutral']
sentiment_label = text_df.sentiment.factorize()
sentiment_label
text = text_df.text.values
tokenizer = Tokenizer(num_words=5000)



new_text = []
for t in text:
    if type(t) == str:
        t.lower()
    
        # Remove urls from the comments
        t = re.sub(r"http\S+|www\S+|https\S+", '', t, flags=re.MULTILINE)
        # Remove user related references from the comments:: '@' and '#' 
        t = re.sub(r'\@\w+|\#','', t)    
        # Remove punctuations from the comments
        t = t.translate(str.maketrans('', '', string.punctuation))    
        # Remove numbers
        table = str.maketrans('', '', digits)
        t = t.translate(table)
        #Remove emojis
        t = clean(t, no_emoji=True)    
        # Remove stopwords from the comments
        text_tokens = word_tokenize(t)
        filtered_words = [w for w in text_tokens if not w in stop_words]
        joined_text = " ".join(filtered_words)
        new_text.append(joined_text)
    
tokenizer.fit_on_texts(new_text)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(new_text)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
# print(tokenizer.word_index)
# print(new_text[0])
# print(encoded_docs[0])
# print(padded_sequence[0])
embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary()) 
history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=10, batch_size=64)
model.save("senti_model.h5", history)
print("model created")


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])





