import os
from src.data_preprocess import *
from src.NLP_classifier import generate_NB_model
import pickle

def NB_classify_sentence(sentence):
    """
    Args:
        sentence (string): the job description to classify
    Returns:
        string: if the description is fraudulent or not
    """
    if not os.path.exists('results/NB_classifier'):
        generate_NB_model()
    model = pickle.load(open('results/NB_classifier', 'rb'))
    vectorizer = pickle.load(open('results/NB_vectorizer', 'rb'))
    tokens = tokenization_and_filtering_sentence(sentence)
    s = vectorizer.transform([' '.join(tokens)])
    result = model.predict(s)[0]
    if result == 0:
        return "The job description is safe"
    return "The job description is fraudulent"


