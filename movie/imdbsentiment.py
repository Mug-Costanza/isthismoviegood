import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from imdb import IMDb
ia = IMDb()
theMatrix = ia.get_movie('0133093')

import os
print(os.listdir("aclImdb/"))
import warnings
warnings.filterwarnings('ignore')

#importing the training data
imdb_data=pd.read_csv('IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)

#Summary of the dataset
imdb_data.describe()

#sentiment count
imdb_data['sentiment'].value_counts()

#split the dataset
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)

# Lowercasing
imdb_data['review'] = imdb_data['review'].apply(lambda x: x.lower())

# Remove special characters and numbers
imdb_data['review'] = imdb_data['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Tokenization
imdb_data['review'] = imdb_data['review'].apply(lambda x: word_tokenize(x))

# Removing stopwords
stop_words = set(stopwords.words('english'))
imdb_data['review'] = imdb_data['review'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming or Lemmatization
lemmatizer = WordNetLemmatizer()
imdb_data['review'] = imdb_data['review'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Joining tokens back into sentences
imdb_data['review'] = imdb_data['review'].apply(lambda x: ' '.join(x))
