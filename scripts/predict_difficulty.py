"""
Simple CLI to load the saved difficulty model and predict labels for input text.
Usage:
    python scripts/predict_difficulty.py "Title text" "Body text"
"""
import os
import sys
import joblib
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "difficulty_model.joblib")


def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def load_model(path=MODEL_PATH):
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["label_encoder"]


def predict(title, body):
    text = strip_html((title or "") + "\n" + (body or ""))
    pipeline, le = load_model()
    pred_enc = pipeline.predict([text])[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    return pred_label


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_difficulty.py \"Title\" \"Body\"")
        sys.exit(1)
    title = sys.argv[1]
    body = sys.argv[2] if len(sys.argv) > 2 else ""
    label = predict(title, body)
    print(label)
