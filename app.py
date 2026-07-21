import streamlit as st
import os
import re
import joblib
import json
import pandas as pd
import numpy as np
import base64

THIS_DIR = os.path.dirname(__file__)
DIFF_MODEL_PATH = os.path.join(THIS_DIR, "models", "difficulty_model.joblib")
TAG_MODEL_PATH = os.path.join(THIS_DIR, "models", "tag_model.joblib")
THRESHOLDS_PATH = os.path.join(THIS_DIR, 'models', 'thresholds.json')


def load_thresholds(path=THRESHOLDS_PATH):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_thresholds(th, path=THRESHOLDS_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(th, f, indent=2)
        return True
    except Exception:
        return False


def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


@st.cache_resource
def load_difficulty_model(path=DIFF_MODEL_PATH):
    if not os.path.exists(path):
        return None, None
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["label_encoder"]


@st.cache_resource
def load_tag_model(path=TAG_MODEL_PATH):
    if not os.path.exists(path):
        return None
    artifact = joblib.load(path)
    return artifact


def predict_difficulty(pipeline, le, title, body):
    text = strip_html((title or "") + "\n" + (body or ""))
    pred_enc = pipeline.predict([text])[0]
    return le.inverse_transform([pred_enc])[0]


def predict_tags(tag_artifact, title, body):
    text = strip_html((title or "") + "\n" + (body or ""))
    tfidf = tag_artifact["tfidf"]
    clf = tag_artifact["classifier"]
    tag_names = tag_artifact["tag_names"]
    X = tfidf.transform([text])
    # Try to obtain scores (decision_function or predict_proba). Fall back to binary predict.
    scores = None
    if hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function(X)
        except Exception:
            scores = None
    if scores is None and hasattr(clf, "predict_proba"):
        try:
            scores = clf.predict_proba(X)
        except Exception:
            scores = None
    if scores is None:
        pred = clf.predict(X)
        pred_arr = pred.toarray().ravel()
        scores = pred_arr.astype(float)
    # ensure 1D array
    arr = np.array(scores).ravel()
    return list(zip(tag_names, arr))


def top_k_from_scores(name_score_pairs, k=3, threshold=0.0):
    pairs = [(n, float(s)) for n, s in name_score_pairs]
    # convert decision scores to probabilities via sigmoid if values not in [0,1]
    vals = np.array([p for _, p in pairs])
    if vals.max() > 1.0 or vals.min() < 0.0:
        probs = 1.0 / (1.0 + np.exp(-vals))
    else:
        probs = vals
    pairs_prob = [(pairs[i][0], float(probs[i])) for i in range(len(pairs))]
    filtered = [p for p in pairs_prob if p[1] >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:k]


def apply_per_tag_thresholds(name_score_pairs, per_tag_thresholds, global_threshold=0.0, k=3):
    # name_score_pairs: list of (name, score)
    pairs = [(n, float(s)) for n, s in name_score_pairs]
    vals = np.array([p for _, p in pairs])
    if vals.size and (vals.max() > 1.0 or vals.min() < 0.0):
        probs = 1.0 / (1.0 + np.exp(-vals))
    else:
        probs = vals
    pairs_prob = [(pairs[i][0], float(probs[i])) for i in range(len(pairs))]
    filtered = []
    for name, prob in pairs_prob:
        thr = per_tag_thresholds.get(name, global_threshold)
        if prob >= thr:
            filtered.append((name, prob))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:k]



st.set_page_config(page_title="SO Tag & Difficulty Predictor", layout="centered")
st.title("StackOverflow Tag & Difficulty Predictor")
st.markdown("Enter a question `Title` and `Body` to see predicted tags and difficulty.")

# Sidebar controls and examples
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K tags", 1, 10, 3)
    threshold = st.slider("Tag probability threshold", 0.0, 1.0, 0.2, 0.01)
    st.markdown("---")
    st.markdown("**Examples**")
    ex = st.selectbox("Choose an example", [
        "-- none --",
        "Java Optional immutability",
        "Install React error",
        "Python asyncio await vs yield",
        "CSS center div horizontally",
        "SQL group by performance",
        "Docker permission denied on npm install",
    ]) 
    st.markdown('---')
    st.subheader('Per-tag thresholds')
    enable_per_tag = st.checkbox('Enable per-tag thresholds', value=False)
    # load thresholds from disk into session_state if not present
    if 'per_tag_thresholds' not in st.session_state:
        st.session_state['per_tag_thresholds'] = load_thresholds()
    per_tag_thresholds = st.session_state['per_tag_thresholds']
    model_available = os.path.exists(TAG_MODEL_PATH)
    if enable_per_tag and model_available:
        try:
            # load tag names from model artifact to populate options
            art = joblib.load(TAG_MODEL_PATH)
            available_tags = list(art.get('tag_names', []))[:400]
        except Exception:
            available_tags = []
        sel = st.multiselect('Select tags to set custom threshold', options=available_tags, default=list(per_tag_thresholds.keys()))
        for t in sel:
            cur = per_tag_thresholds.get(t, threshold)
            v = st.slider(f'Threshold for {t}', 0.0, 1.0, float(cur), 0.01)
            per_tag_thresholds[t] = float(v)
        # remove deselected
        for t in list(per_tag_thresholds.keys()):
            if t not in sel:
                per_tag_thresholds.pop(t, None)
        st.session_state['per_tag_thresholds'] = per_tag_thresholds
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button('Save thresholds'):
                ok = save_thresholds(per_tag_thresholds)
                if ok:
                    st.success('Thresholds saved to models/thresholds.json')
                else:
                    st.error('Failed to save thresholds')
        with col2:
            if st.button('Reset thresholds'):
                st.session_state['per_tag_thresholds'] = {}
                try:
                    if os.path.exists(THRESHOLDS_PATH):
                        os.remove(THRESHOLDS_PATH)
                except Exception:
                    pass
                st.experimental_rerun()
    elif enable_per_tag and not model_available:
        st.info('Tag model not available yet — train or place `tag_model.joblib` in models/')

diff_pipeline, diff_le = load_difficulty_model()
tag_artifact = load_tag_model()

st.session_state.setdefault('last_prediction', None)
st.session_state.setdefault('last_out_text', '')
with st.form("single_predict"):
    title = st.text_input("Title")
    body = st.text_area("Body (HTML allowed)")
    if ex == "Java Optional immutability":
        title = title or "Why are Java Optionals immutable?"
        body = body or "I'd like to understand why Java 8 Optionals were designed to be immutable. Is it just for thread-safety?"
    elif ex == "Install React error":
        title = title or "I'm unable to install new react"
        body = body or "I tried npm install but get permission errors and the package doesn't install."
    elif ex == "Python asyncio await vs yield":
        title = title or "What's the difference between await and yield in Python?"
        body = body or "I am learning asyncio and want to know when to use await vs yield from generators. Examples would help."
    elif ex == "CSS center div horizontally":
        title = title or "How to center a div horizontally?"
        body = body or "I want to center a child div inside a parent using CSS. What are the simplest ways?"
    elif ex == "SQL group by performance":
        title = title or "Improve GROUP BY performance"
        body = body or "My query with GROUP BY is slow on large tables. What indexing or rewrite strategies can help?"
    elif ex == "Docker permission denied on npm install":
        title = title or "Permission denied during npm install in Docker"
        body = body or "When running npm install in a Dockerfile I see EACCES permission denied errors. How to fix?"
    submitted = st.form_submit_button("Predict")
    if submitted:
        if diff_pipeline is not None:
            # difficulty probability
            try:
                probs = diff_pipeline.predict_proba([strip_html((title or "") + "\n" + (body or ""))])[0]
                classes = list(diff_le.classes_)
                class_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                st.success(f"Predicted difficulty: {class_probs[0][0]} ({class_probs[0][1]:.2f})")
                st.write("Difficulty probabilities:")
                for c, p in class_probs:
                    st.write(f"- {c}: {p:.3f}")
            except Exception:
                label = predict_difficulty(diff_pipeline, diff_le, title, body)
                st.success(f"Predicted difficulty: {label}")
        else:
            st.warning("Difficulty model not available.")

        if tag_artifact is not None:
            name_scores = predict_tags(tag_artifact, title, body)
            per_tag_thresholds = st.session_state.get('per_tag_thresholds', {}) if enable_per_tag else {}
            top = apply_per_tag_thresholds(name_scores, per_tag_thresholds, global_threshold=threshold, k=top_k)
            if top:
                    st.markdown("**Top tags**")
                    cols = st.columns(min(len(top), 5))
                    for i, (n, s) in enumerate(top):
                        with cols[i % len(cols)]:
                            st.markdown(f"**{n}**")
                            try:
                                st.progress(min(max(s, 0.0), 1.0))
                            except Exception:
                                st.write(f"{s:.3f}")
                            st.caption(f"p={s:.3f}")
            else:
                st.info("(no tags above threshold)")

            # Explanation toggle
            explain = st.checkbox("Show token-level explanations")
            if explain:
                st.markdown("**Explanations**")
                # Difficulty explanations
                try:
                    diff_tfidf = diff_pipeline.named_steps['tfidf']
                    diff_clf = diff_pipeline.named_steps['clf']
                    text_vec = diff_tfidf.transform([strip_html((title or "") + "\n" + (body or ""))])
                    pred_class = None
                    if hasattr(diff_clf, 'coef_'):
                        if hasattr(diff_pipeline, 'predict'):
                            pred_class = diff_pipeline.predict([strip_html((title or "") + "\n" + (body or ""))])[0]
                        classes = list(diff_le.classes_)
                        class_idx = classes.index(pred_class) if pred_class in classes else 0
                        coef = diff_clf.coef_[class_idx]
                        feature_names = diff_tfidf.get_feature_names_out()
                        contrib = (text_vec.toarray().ravel() * coef)
                        top_idx = np.argsort(contrib)[-10:][::-1]
                        st.write(f"Top tokens for difficulty {pred_class}:")
                        st.write([f"{feature_names[i]} ({contrib[i]:.4f})" for i in top_idx if contrib[i] != 0])
                except Exception:
                    st.write("Difficulty explanations not available for this model.")

                # Tag explanations
                try:
                    tfidf = tag_artifact['tfidf']
                    clf = tag_artifact['classifier']
                    tag_names = tag_artifact['tag_names']
                    text_vec_tag = tfidf.transform([strip_html((title or "") + "\n" + (body or ""))])
                    # clf may be OneVsRestClassifier
                    estimators = getattr(clf, 'estimators_', None)
                    if estimators is None:
                        st.write('Tag model does not expose estimators for explanations.')
                    else:
                        for tag, _ in top:
                            try:
                                idx = list(tag_names).index(tag)
                            except ValueError:
                                continue
                            est = estimators[idx]
                            if hasattr(est, 'coef_'):
                                coef = est.coef_.ravel()
                                feature_names = tfidf.get_feature_names_out()
                                contrib = (text_vec_tag.toarray().ravel() * coef)
                                top_idx = np.argsort(contrib)[-10:][::-1]
                                st.write(f"Top tokens for tag '{tag}':")
                                st.write([f"{feature_names[i]} ({contrib[i]:.4f})" for i in top_idx if contrib[i] != 0])
                except Exception:
                    st.write("Tag explanations not available for this model.")

            # option to show full tag probabilities and search
            show_all = st.checkbox("Show full tag probabilities")
            if show_all and tag_artifact is not None:
                try:
                    name_scores = predict_tags(tag_artifact, title, body)
                    df_probs = pd.DataFrame(name_scores, columns=['tag', 'score'])
                    df_probs['score'] = df_probs['score'].astype(float)
                    q = st.text_input('Filter tags (contains)')
                    if q:
                        df_show = df_probs[df_probs['tag'].str.contains(q, case=False, na=False)].sort_values('score', ascending=False)
                    else:
                        df_show = df_probs.sort_values('score', ascending=False)
                    st.dataframe(df_show.reset_index(drop=True))
                except Exception:
                    st.write('Failed to show tag probabilities.')

            # prepare result and store in session_state so downloads can be shown outside the form
            out_text = "Predicted difficulty: " + (class_probs[0][0] if diff_pipeline is not None else "N/A") + "\n"
            out_text += "Predicted tags: " + (", ".join([n for n, _ in top]) if top else "")
            result_json = {
                'title': title,
                'body': body,
                'predicted_tags': [n for n, _ in top] if tag_artifact is not None else [],
            }
            if diff_pipeline is not None:
                try:
                    probs = diff_pipeline.predict_proba([strip_html((title or "") + "\n" + (body or ""))])[0]
                    classes = list(diff_le.classes_)
                    result_json['difficulty_probs'] = {c: float(p) for c, p in zip(classes, probs)}
                    result_json['predicted_difficulty'] = sorted(result_json['difficulty_probs'].items(), key=lambda x: x[1], reverse=True)[0][0]
                except Exception:
                    result_json['predicted_difficulty'] = predict_difficulty(diff_pipeline, diff_le, title, body)
            if tag_artifact is not None:
                name_scores = predict_tags(tag_artifact, title, body)
                result_json['tag_scores'] = {n: float(s) for n, s in name_scores}
            st.session_state['last_prediction'] = result_json
            st.session_state['last_out_text'] = out_text
        else:
            st.warning("Tag model not available. Run training script to create it.")

        # JSON export handled via session_state after form
        pass

# After the form: show download/copy buttons for last prediction (allowed outside forms)
if st.session_state.get('last_prediction'):
    last = st.session_state['last_prediction']
    out_text = st.session_state.get('last_out_text', '')
    try:
        st.download_button("Download result", out_text, file_name="prediction.txt", mime="text/plain")
    except Exception:
        pass
    # copy to clipboard using small JS
    try:
        copy_html = f"""
        <input type='text' value='{out_text}' id='toCopy' style='position: absolute; left: -1000px;' />
        <button onclick="navigator.clipboard.writeText(document.getElementById('toCopy').value)">Copy to clipboard</button>
        """
        st.components.v1.html(copy_html)
    except Exception:
        pass
    try:
        st.download_button("Download JSON result", json.dumps(last, indent=2), file_name='prediction.json', mime='application/json')
    except Exception:
        pass

st.markdown("---")
st.header("Batch CSV Prediction")
st.markdown("Upload a CSV with `Title` and `Body` columns to append `predicted_difficulty` and `predicted_tags` and download the result.")
uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None
    if df is not None:
        if not all(col in df.columns for col in ["Title", "Body"]):
            st.warning("CSV should contain `Title` and `Body` columns.")
        else:
            if st.button("Add predictions and prepare download"):
                texts = (df["Title"].fillna("") + "\n" + df["Body"].fillna(""))
                if diff_pipeline is not None:
                    df["predicted_difficulty"] = [
                        (lambda t: (
                            (lambda probs, classes: sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[0][0])(
                                diff_pipeline.predict_proba([strip_html(t)])[0], list(diff_le.classes_)
                            )
                        ))(t)
                        for t in texts
                    ]
                if tag_artifact is not None:
                    tfidf = tag_artifact["tfidf"]
                    clf = tag_artifact["classifier"]
                    tag_names = tag_artifact["tag_names"]
                    X = tfidf.transform(texts.tolist())
                    # get scores
                    if hasattr(clf, "decision_function"):
                        try:
                            scores = clf.decision_function(X)
                        except Exception:
                            scores = None
                    else:
                        scores = None
                    if scores is None and hasattr(clf, "predict_proba"):
                        try:
                            scores = clf.predict_proba(X)
                        except Exception:
                            scores = None
                    tag_lists = []
                    for i in range(X.shape[0]):
                        if scores is None:
                            pred = clf.predict(X[i])
                            arr = pred.toarray().ravel()
                        else:
                            arr = np.array(scores[i]).ravel()
                        # convert to probs
                        if arr.max() > 1.0 or arr.min() < 0.0:
                            probs = 1.0 / (1.0 + np.exp(-arr))
                        else:
                            probs = arr
                        per_tag_thresholds = st.session_state.get('per_tag_thresholds', {})
                        top_candidates = []
                        for n, p in zip(tag_names, probs):
                            thr = per_tag_thresholds.get(n, threshold)
                            if p >= thr:
                                top_candidates.append((n, float(p)))
                        top = sorted(top_candidates, key=lambda x: x[1], reverse=True)[:top_k]
                        tag_lists.append(' '.join([n for n, _ in top]))
                    df['predicted_tags'] = tag_lists

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download augmented CSV", csv_bytes, file_name="with_predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown("Difficulty model: "+DIFF_MODEL_PATH)
st.markdown("Tag model: "+TAG_MODEL_PATH)

# Diagnostics
st.markdown("---")
st.header("Diagnostics")
metrics_path = os.path.join(THIS_DIR, 'models', 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        st.subheader('Tag model metrics')
        tags_metrics = metrics.get('tags', {}).get('per_tag', {})
        if tags_metrics:
            # show top 10 by f1
            items = [(t, v['f1']) for t, v in tags_metrics.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            st.write('Top 10 tags by F1')
            for t, f1 in items[:10]:
                st.write(f"- {t}: f1={f1:.3f}")
        else:
            st.write('No tag metrics found.')
    except Exception as e:
        st.write('Failed to load metrics:', e)
else:
    st.write('No metrics.json found — run `scripts/evaluate_models.py` to generate model diagnostics.')

# Model metadata download (zip models dir)
try:
    import io, zipfile
    model_files = []
    models_dir = os.path.join(THIS_DIR, 'models')
    if os.path.isdir(models_dir):
        for fn in os.listdir(models_dir):
            model_files.append(os.path.join(models_dir, fn))
    if model_files:
        if st.button('Prepare model metadata ZIP'):
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, mode='w') as z:
                for p in model_files:
                    z.write(p, arcname=os.path.basename(p))
            bio.seek(0)
            st.download_button('Download models ZIP', data=bio.read(), file_name='models_bundle.zip', mime='application/zip')
    else:
        st.write('No model artifacts to bundle in `models/`.')
except Exception as e:
    st.write('Model bundle feature unavailable:', e)
