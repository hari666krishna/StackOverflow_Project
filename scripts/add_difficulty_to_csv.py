"""
Load an input CSV with `Title` and `Body` columns, predict difficulty using saved model,
and write an output CSV with an added `predicted_difficulty` column.

Usage:
    python scripts/add_difficulty_to_csv.py data/input.csv data/output_with_difficulty.csv
"""
import os
import sys
import joblib
import pandas as pd
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "difficulty_model.joblib")


def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def load_model(path=MODEL_PATH):
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["label_encoder"]


def predict_series(df, pipeline, le):
    texts = (df["Title"].fillna("") + "\n" + df["Body"].fillna("")).apply(strip_html)
    preds_enc = pipeline.predict(texts)
    preds = le.inverse_transform(preds_enc)
    return preds


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/add_difficulty_to_csv.py input.csv output.csv")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    df = pd.read_csv(inp)
    pipeline, le = load_model()
    df["predicted_difficulty"] = predict_series(df, pipeline, le)
    df.to_csv(out, index=False)
    print(f"Wrote augmented CSV to {out}")

if __name__ == "__main__":
    main()
