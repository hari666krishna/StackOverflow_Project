"""
Train tag prediction model and save artifact at models/tag_model.joblib
Usage: python scripts/train_tag_model.py
"""
import os
import re
import joblib
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, classification_report

ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(ROOT, '..', 'StackOverflow-Tag-Prediction')
TRAIN_P = os.path.join(PROJECT_ROOT, 'preprocessedTrain.csv')
TEST_P = os.path.join(PROJECT_ROOT, 'preprocessedTest.csv')
MODELS_DIR = os.path.join(ROOT, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
OUT_PATH = os.path.join(MODELS_DIR, 'tag_model.joblib')


def cleaning(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    pattern = re.compile(r'http[s]?://\S+')
    clean = re.compile(r'<.*?>')
    text = re.sub(clean, '', text)
    text = pattern.sub('', text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def load_data(path):
    df = pd.read_csv(path)
    df['Title'] = df['Title'].fillna('').map(cleaning)
    df['Body'] = df['Body'].fillna('').map(cleaning)
    df['text'] = df['Title'] + ' ' + df['Body']
    df = df[df['Tags'].notna()]
    return df


def main():
    if not os.path.exists(TRAIN_P) or not os.path.exists(TEST_P):
        raise FileNotFoundError('Run scripts/preprocess.py first to create preprocessed files')

    train = load_data(TRAIN_P)
    test = load_data(TEST_P)

    vectorizerTF = TfidfVectorizer(min_df=0.00009, max_features=10000, smooth_idf=True, norm='l2', tokenizer=str.split, ngram_range=(1,3))
    X_train = vectorizerTF.fit_transform(train['text'])
    X_test = vectorizerTF.transform(test['text'])

    vectorizerCV = CountVectorizer(tokenizer=str.split, binary=True, max_features=1500)
    y_train = vectorizerCV.fit_transform(train['Tags'])
    y_test = vectorizerCV.transform(test['Tags'])

    clf = OneVsRestClassifier(SGDClassifier(loss='log_loss', alpha=1e-5, penalty='l1'), n_jobs=-1)
    print('Training tag classifier...')
    start = datetime.now()
    clf.fit(X_train, y_train)
    print('Training done in', datetime.now() - start)

    preds = clf.predict(X_test)
    print('accuracy :', accuracy_score(y_test, preds))
    print('macro f1 :', f1_score(y_test, preds, average='macro'))
    print('micro f1 :', f1_score(y_test, preds, average='micro'))
    print('hamming loss :', hamming_loss(y_test, preds))

    artifact = {
        'classifier': clf,
        'tfidf': vectorizerTF,
        'tag_vectorizer': vectorizerCV,
        'tag_names': vectorizerCV.get_feature_names_out(),
    }
    joblib.dump(artifact, OUT_PATH)
    print('Saved tag model to', OUT_PATH)


if __name__ == '__main__':
    main()
