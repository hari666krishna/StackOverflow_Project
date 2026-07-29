# StackOverflow Tag & Difficulty Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/hari666krishna/StackOverflow_Project)

This project predicts StackOverflow question tags and a difficulty label.

Quick start (local, using the provided venv):

1. Activate the virtual environment (Windows PowerShell):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& .\venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install -r StackOverflow-Tag-Prediction/requirements.txt
```

3. (Optional) Train models from raw CSVs:

```powershell
python StackOverflow-Tag-Prediction/scripts/preprocess.py
python StackOverflow-Tag-Prediction/scripts/train_tag_model.py
python StackOverflow-Tag-Prediction/scripts/train_difficulty.py
```

4. Run the Streamlit app:

```powershell
python -m streamlit run StackOverflow-Tag-Prediction/app.py
```

New features added by the assistant:

- Per-tag threshold editor: open the sidebar, enable "Per-tag thresholds", select tags and set custom thresholds, then click "Save thresholds" to persist to `models/thresholds.json`.
- Diagnostics: run `scripts/evaluate_models.py` to produce `models/metrics.json` and plots (e.g. `models/tags_top_f1.png`).
- Export: download model artifacts as a ZIP from the Diagnostics page.

App features:

- Single-item prediction (Title + Body): shows predicted difficulty and tags.
- Batch CSV upload: append `predicted_difficulty` and `predicted_tags` and download.

Deployment (Docker):

```bash
docker build -t so-predictor .
docker run -p 8501:8501 so-predictor
```

Model artifacts are stored in `models/` after training. The app expects
`models/difficulty_model.joblib` and `models/tag_model.joblib` to exist.

If you want improvements, I can:

- add thresholding / top-k for tag outputs
- improve tag model training (class balancing, label pruning)
- add tests and CI

# Stack Overflow Tag Prediction 🏷️

`A machine learning model that predicts tags for a given question and body.`

<p align="center">
  <img src="https://github.com/Ankit152/StackOverflow-Tag-Prediction/blob/main/img/so-logo.jpg" >
</p>

**Dataset Link:** https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate

## For developers, by developers 👨‍💻

Stack Overflow is an open community for anyone that codes. They help you get answers to your toughest coding questions, share knowledge with your coworkers in private, and find your next dream job.

## For businesses, by developers 🕴️

Their mission is to help developers write the script of the future. This means helping you find and hire skilled developers for your business and providing them the tools they need to share knowledge and work effectively.

### Problem Defination 🤔

Given a `Title` and the `Body` of a question, we have to predict the relevant tags such that the question gets recommended to the `right domain expert` so that the expert can `answer the question correctly`.

### Business Constraints ✔️

- To predict as many tags as possible with very high `precision` and `recall`.
- `Incorrect tags` could impact the `customer experience` on Stack Overflow.
- No strict latency constraints. The model should be able to generate the relevant tags in a `reasonable` amount of `time`.

### Data 🗄️

- `train.csv` = 48 MB
- `test.csv` = 16 MB

The data consists of 6 columns.

1. Id: Represents the ID of the question
2. Title: Represents the title of the question
3. Body: Represents the body of the question where the question is explained properly
4. Tags: The tags relevant for the question asked
5. CreationDate: The date at which the question was asked
6. Type: Deals with the quality of the question

Our main important features in the dataset are `Title`,`Body` and `Tags`.

## Plots for better understanding 📊

### Countplot of Tags per question 📈

_`This is the countplot of number of tags per question.`_

<p align="center">
  <img src="https://github.com/Ankit152/StackOverflow-Tag-Prediction/blob/main/img/tagCount.jpg" height=612>
</p>

The key take away from the above plot is that most of the question has `2` or `3` tags in them.

### Distribution of Tags 📉

_`This is the distribution of number of times the tag appeared in questions.`_

<p align="center">
  <img src="https://github.com/Ankit152/StackOverflow-Tag-Prediction/blob/main/img/tagDistribution.jpg" height=612>
</p>

The key take away from the above plot is that a tag is appearing 5 time in max.

### WordCloud ☁️

_`This is the wordcloud generated from the tags and it's count.`_
## Author 
Sanneboina HariKrishna

B.Tech – Computer Science & Engineering (AI & ML)

Machine Learning Project – StackOverflow-Tag-Prediction


