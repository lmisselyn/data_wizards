import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import nltk
import string
from data_preprocess import *
import pickle

def transformer_classify_sentence(sentence):
    model = BertForSequenceClassification.from_pretrained('./results/checkpoint-5364')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()

    sentence = ' '.join(tokenization_and_filtering_sentence(sentence))
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

def NB_classify_sentence(sentence):
    model = pickle.load(open('results/NB_classifier', 'rb'))
    vectorizer = pickle.load(open('results/NB_vectorizer', 'rb'))
    tokens = tokenization_and_filtering_sentence(sentence)
    s = vectorizer.transform([' '.join(tokens)])
    return model.predict(s)[0]


if __name__ == "__main__":
    # df = pd.read_csv("Datasets/cleaned_dataset.csv")
    # X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)
    # y_pred = []
    # for sentence in X_test:
    #     y_pred.append(transformer_classify_sentence(sentence))
    # f1 = f1_score(y_test, y_pred, average='binary')
    # print(f"f1 score : {f1}")
    print(transformer_classify_sentence('ok lq'))
    print(NB_classify_sentence('ok lq'))
