"""
Evaluate tag and difficulty models on preprocessed test data.
Produces: models/metrics.json
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT = os.path.join(ROOT, 'StackOverflow-Tag-Prediction')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(MODELS, exist_ok=True)

PREP_TEST = os.path.join(PROJECT, 'preprocessedTest.csv')
DIFF_MODEL = os.path.join(MODELS, 'difficulty_model.joblib')
TAG_MODEL = os.path.join(MODELS, 'tag_model.joblib')
METRICS_OUT = os.path.join(MODELS, 'metrics.json')
DIFF_CONF_PNG = os.path.join(MODELS, 'confusion_difficulty.png')
TAGS_F1_PNG = os.path.join(MODELS, 'tags_top_f1.png')


def load_data(path):
    df = pd.read_csv(path)
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    return df


def eval_difficulty(df, path):
    art = joblib.load(path)
    pipeline = art['pipeline']
    le = art['label_encoder']
    if 'Y' not in df.columns:
        return None
    y_true = df['Y']
    X = df['text'].tolist()
    y_pred_enc = pipeline.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # confusion matrix plot
    try:
        labels = list(le.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Difficulty confusion matrix')
        plt.tight_layout()
        plt.savefig(DIFF_CONF_PNG)
        plt.close()
        report['_confusion_png'] = os.path.basename(DIFF_CONF_PNG)
    except Exception:
        pass
    return report


def eval_tags(df, path):
    art = joblib.load(path)
    tfidf = art['tfidf']
    clf = art['classifier']
    tag_vec = art['tag_vectorizer']
    tag_names = list(art['tag_names'])
    X = tfidf.transform(df['text'].tolist())
    y_true = tag_vec.transform(df['Tags'])
    y_pred = clf.predict(X)
    # per-tag classification
    from sklearn.metrics import precision_recall_fscore_support
    y_true_arr = y_true.toarray()
    y_pred_arr = y_pred.toarray()
    per_tag = {}
    for i, tag in enumerate(tag_names):
        p, r, f, _ = precision_recall_fscore_support(y_true_arr[:, i], y_pred_arr[:, i], average='binary', zero_division=0)
        per_tag[tag] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}
    # save top-N tags by f1 plot
    try:
        items = [(t, v['f1']) for t, v in per_tag.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:20]
        tags, f1s = zip(*top) if top else ([], [])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(f1s), y=list(tags), palette='viridis')
        plt.xlabel('F1 score')
        plt.title('Top 20 tags by F1')
        plt.tight_layout()
        plt.savefig(TAGS_F1_PNG)
        plt.close()
        return {'per_tag': per_tag, '_top_f1_png': os.path.basename(TAGS_F1_PNG)}
    except Exception:
        return {'per_tag': per_tag}
    return {'per_tag': per_tag}


def main():
    df = load_data(PREP_TEST)
    metrics = {}
    if os.path.exists(DIFF_MODEL):
        try:
            metrics['difficulty'] = eval_difficulty(df, DIFF_MODEL)
        except Exception as e:
            metrics['difficulty_error'] = str(e)
    else:
        metrics['difficulty'] = None

    if os.path.exists(TAG_MODEL):
        try:
            metrics['tags'] = eval_tags(df, TAG_MODEL)
        except Exception as e:
            metrics['tags_error'] = str(e)
    else:
        metrics['tags'] = None

    with open(METRICS_OUT, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Wrote metrics to', METRICS_OUT)


if __name__ == '__main__':
    main()
