import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
from src.data_preprocess import *

def simple_NB_classifier(df):
    """
    Args:
        df (pd.Dataframe): the dataframe containing the labbelled job description
    Train a simple Naive Bayes classifier and store it in a pickle file
    """
    vectorizer = TfidfVectorizer()
    preprocessed_text = []
    tokens = tokenization_and_filtering_dataframe(df)
    for t in tokens:
        preprocessed_text.append(' '.join(t))
    y = df['label']
    vectorizer.fit(preprocessed_text)
    if not os.path.exists('results'):
        os.makedirs('results')
    pickle.dump(vectorizer, open('results/NB_vectorizer', 'wb'))
    X = vectorizer.transform(preprocessed_text)
    
    #Stratified split to ensure proportionnality 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) 
    #Synthetic Minority Oversampling Technique to adress dataset imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train= smote.fit_resample(X_train, y_train)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    pickle.dump(model, open('results/NB_classifier', 'wb'))
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")
    
def generate_NB_model():
    filename = 'datasets/cleaned_dataset.csv'
    df = pd.read_csv(filename)
    simple_NB_classifier(df)

