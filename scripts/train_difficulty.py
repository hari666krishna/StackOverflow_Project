"""
Train a question difficulty classifier and save the pipeline.
Usage: python scripts/train_difficulty.py

Outputs:
- models/difficulty_model.joblib (contains vectorizer, classifier, label encoder)
"""
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VALID_CSV = os.path.join(DATA_DIR, "valid.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def strip_html(text):
    if not isinstance(text, str):
        return ""
    # remove HTML tags
    return re.sub(r"<[^>]+>", " ", text)


def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: Title, Body, Y
    df["Title"] = df["Title"].fillna("")
    df["Body"] = df["Body"].fillna("")
    df["text"] = (df["Title"] + " \n" + df["Body"]).apply(strip_html)
    df = df[df["Y"].notna()]
    return df


def main():
    if os.path.exists(VALID_CSV):
        train_df = load_data(TRAIN_CSV)
        valid_df = load_data(VALID_CSV)
        X_train = train_df["text"]
        y_train = train_df["Y"]
        X_valid = valid_df["text"]
        y_valid = valid_df["Y"]
    else:
        df = load_data(TRAIN_CSV)
        X_train, X_valid, y_train, y_valid = train_test_split(
            df["text"], df["Y"], test_size=0.2, random_state=42, stratify=df["Y"]
        )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_valid_enc = le.transform(y_valid)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print("Training difficulty classifier...")
    pipeline.fit(X_train, y_train_enc)

    preds = pipeline.predict(X_valid)
    print(classification_report(y_valid_enc, preds, target_names=le.classes_))

    artifact = {
        "pipeline": pipeline,
        "label_encoder": le,
    }
    out_path = os.path.join(MODELS_DIR, "difficulty_model.joblib")
    joblib.dump(artifact, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
