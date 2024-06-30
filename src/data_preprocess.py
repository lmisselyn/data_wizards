import pandas as pd
import numpy as np
from cleantext import clean
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import csv
import nltk
import string
import spacy
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

def handle_bad_row(row):
    text = ' '.join(row[:-1])
    return [text, row[-1]]

def preprocess_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONE)
        data = [row for row in reader]
    
    texts = []
    labels = []

    for row in data[1:]:
        if len(row) != 2:
            row = handle_bad_row(row)
        text, label = row
        text = text.strip()
        text = text = " ".join(text.split())
        if text == "":
            text = " "
        texts.append(text)
        labels.append(int(label.strip()))
        
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

def tokenization_and_filtering_dataframe(df):
    tokenized = []
    for i in range(len(df)):
        line = df['text'][i]
        if type(line) is not str:
            print(i)
        tokens = nltk.word_tokenize(line)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in string.punctuation]
        tokenized.append(tokens)
    return tokenized

def tokenization_and_filtering_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens


def preprocess_text(df):
    text = df['text']
    nlp = spacy.load('en_core_web_sm')
    preprocessed_text = []
    for doc in nlp.pipe(text, disable=['parser', 'ner']):
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        preprocessed_text.append(' '.join(tokens))
    return preprocessed_text
        
        