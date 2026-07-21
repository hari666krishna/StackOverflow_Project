import joblib
import re
import os


def strip_html(text):
    return re.sub(r"<[^>]+>", " ", text) if isinstance(text, str) else ""


TITLE = "Why are Java Optionals immutable?"
BODY = "I'd like to understand why Java 8 Optionals were designed to be immutable. Is it just for thread-safety?"
text = strip_html(TITLE + "\n" + BODY)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIFF_PATH = os.path.join(BASE, 'models', 'difficulty_model.joblib')
TAG_PATH = os.path.join(BASE, 'models', 'tag_model.joblib')

# Difficulty model
if not os.path.exists(DIFF_PATH):
    print('Difficulty model not found at', DIFF_PATH)
else:
    diff_art = joblib.load(DIFF_PATH)
    pipeline = diff_art['pipeline']
    le = diff_art['label_encoder']
    pred_enc = pipeline.predict([text])[0]
    pred_diff = le.inverse_transform([pred_enc])[0]
    print('Predicted difficulty:', pred_diff)

# Tag model
if not os.path.exists(TAG_PATH):
    print('Tag model not found at', TAG_PATH)
else:
    tag_art = joblib.load(TAG_PATH)
    tfidf = tag_art['tfidf']
    clf = tag_art['classifier']
    tag_names = tag_art['tag_names']
    X = tfidf.transform([text])
    pred = clf.predict(X)
    pred_arr = pred.toarray().ravel()
    tags = [t for t, v in zip(tag_names, pred_arr) if v == 1]
    print('Predicted tags:', tags)
