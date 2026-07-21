"""
Preprocess raw data (data/train.csv and data/valid.csv) into
preprocessedTrain.csv and preprocessedTest.csv for tag model training.
"""
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VALID_CSV = os.path.join(DATA_DIR, "valid.csv")
OUT_TRAIN = os.path.join(ROOT, "StackOverflow-Tag-Prediction", "preprocessedTrain.csv")
OUT_TEST = os.path.join(ROOT, "StackOverflow-Tag-Prediction", "preprocessedTest.csv")


def clean_tags(s):
    s = str(s)
    ls = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] == '<':
            for j in range(i+1, n):
                if s[j] == '>':
                    ls.append(s[i+1:j])
                    i = j + 1
                    break
        else:
            i += 1
    return " ".join(ls)


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(VALID_CSV)
    train = train.drop(columns=['Id','CreationDate','Y'], errors='ignore')
    test = test.drop(columns=['Id','CreationDate','Y'], errors='ignore')
    train['Tags'] = train['Tags'].apply(clean_tags)
    test['Tags'] = test['Tags'].apply(clean_tags)
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)
    print(f"Wrote {OUT_TRAIN} and {OUT_TEST}")


if __name__ == '__main__':
    main()
