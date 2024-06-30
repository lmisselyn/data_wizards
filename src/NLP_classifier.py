import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_preprocess import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric, Dataset
import torch
import pickle

def simple_NB_classifier(df):
    vectorizer = TfidfVectorizer()
    preprocessed_text = []
    tokens = tokenization_and_filtering_dataframe(df)
    for t in tokens:
        preprocessed_text.append(' '.join(t))
    y = df['label']
    vectorizer.fit(preprocessed_text)
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

def transformer_NLP_classifier(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    preprocessed_text = []
    tokens = tokenization_and_filtering_dataframe(df)
    for t in tokens:
        preprocessed_text.append(' '.join(t))

    encodings = tokenizer(preprocessed_text, truncation=True, padding=True, max_length=128)

    X = pd.DataFrame({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    })
    
    y = pd.Series(df['label'], name='labels')

    # Stratified train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_dataset = Dataset.from_pandas(pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='labels')], axis=1))
    test_dataset = Dataset.from_pandas(pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        precision, recall, f1, _ = classification_report(labels, preds, output_dict=True)['weighted avg'].values()
        accuracy = (preds == labels).mean()
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    trainer = Trainer(
        model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    
    model_save_path = './saved_model'
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    

if __name__=="__main__": 
    filename = 'datasets/cleaned_dataset.csv'
    df = pd.read_csv(filename)
    #transformer_NLP_classifier(df)
    simple_NB_classifier(df)
