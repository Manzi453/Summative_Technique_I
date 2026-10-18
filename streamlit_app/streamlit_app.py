"""Swahili tweet sentiment: a small web page around the model fine-tuned in notebook 4.

Run from the project folder:  streamlit run streamlit_app/streamlit_app.py
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL = ROOT / "models" / "best_sentiment"
DEPLOYED_RECORD = ROOT / "results" / "metrics" / "deployed_model.json"
METRICS_SUMMARY = ROOT / "results" / "metrics_summary.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
MAX_LEN = 128  # same as training
LABEL_COLOUR = {"negative": "red", "neutral": "gray", "positive": "green"}

st.set_page_config(page_title="Swahili tweet sentiment", page_icon="💬", layout="centered")


def clean_text(text):
    """The same cleaning the model was trained on (notebook 1): lower-case, drop links, collapse spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def model_source():
    """MODEL_ID if set, else the local folder from notebook 4, else the Hugging Face repo it was uploaded to."""
    if os.environ.get("MODEL_ID"):
        return os.environ["MODEL_ID"]
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL)
    if DEPLOYED_RECORD.exists():
        return json.loads(DEPLOYED_RECORD.read_text())["hf_repo_id"]
    return None


@st.cache_resource(show_spinner="Loading the model (the first time takes a little while)...")
def load_model(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tokenizer, model, labels


def predict(text, tokenizer, model):
    batch = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.inference_mode():
        return torch.softmax(model(**batch).logits[0], dim=-1).tolist()


@st.cache_data
def load_examples():
    """A few real tweets from the held-out test set: two per class, short enough to read."""
    if not TEST_CSV.exists():
        return []
    df = pd.read_csv(TEST_CSV).dropna(subset=["text_clean"])
    df = df[df["text_clean"].str.split().str.len().between(6, 25)]
    picks = [g.sample(min(2, len(g)), random_state=7) for _, g in df.groupby("label")]
    return [(r.label, r.text_clean) for r in pd.concat(picks).itertuples()]


@st.cache_data
def load_results():
    if not METRICS_SUMMARY.exists():
        return None
    return pd.read_csv(METRICS_SUMMARY)


def use_example():
    choice = st.session_state.example
    if choice is not None:
        label, text = EXAMPLES[choice]
        st.session_state.tweet = text
        st.session_state.example_text, st.session_state.example_label = text, label


EXAMPLES = load_examples()
st.session_state.setdefault("tweet", "")

st.title("Swahili tweet sentiment")
st.caption("Paste a tweet in Swahili and the model says whether it sounds positive, neutral or negative.")

source = model_source()
if source is None:
    st.error("No model found. Run notebook 4 first (it saves models/best_sentiment), or set the MODEL_ID environment variable.")
    st.stop()
tokenizer, model, LABELS = load_model(source)

tab_classify, tab_results, tab_about = st.tabs(["Classify", "Model results", "About"])

with tab_classify:
    if EXAMPLES:
        st.selectbox(
            "Or try a real tweet from the test set",
            options=range(len(EXAMPLES)),
            index=None,
            format_func=lambda i: f"[{EXAMPLES[i][0]}] {EXAMPLES[i][1][:70]}...",
            placeholder="Choose an example",
            key="example",
            on_change=use_example,
        )
    tweet = st.text_area("Tweet", key="tweet", height=130, placeholder="Andika tweet hapa...")

    if st.button("Classify", type="primary"):
        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning("Type or paste a tweet first.")
        else:
            probs = predict(cleaned, tokenizer, model)
            best = max(range(len(probs)), key=probs.__getitem__)
            label = LABELS[best]
            st.subheader(f":{LABEL_COLOUR.get(label, 'blue')}[{label.title()}]  ({probs[best]:.0%})")
            if probs[best] < 0.5:
                st.caption("The model is not sure about this one.")
            for name, p in zip(LABELS, probs):
                st.progress(p, text=f"{name.title()}: {p:.1%}")
            if st.session_state.get("example_text") == tweet:
                st.caption(f"The human label for this test tweet is **{st.session_state.example_label}**.")
            with st.expander("Text the model saw"):
                st.write(cleaned)

with tab_results:
    results = load_results()
    if results is None:
        st.info("No results yet. Run notebooks 2 to 4 to create results/metrics_summary.csv.")
    else:
        st.write("All models on the same held-out test set. Macro-F1 is the main score: it averages the three classes, "
                 "so always guessing the biggest class (neutral) does not look good.")
        table = results[["model_name", "accuracy", "macro_f1", "weighted_f1"]].round(3).set_index("model_name")
        st.dataframe(table, width="stretch")
        st.bar_chart(table["macro_f1"], horizontal=True)
        st.caption("Rows labelled 'mean of seeds' average several training runs with different random seeds. "
                   "The test set is small (748 tweets, 80 of them negative), so small differences are not reliable.")

with tab_about:
    served = None
    if DEPLOYED_RECORD.exists():
        d = json.loads(DEPLOYED_RECORD.read_text())
        served = f"{d['model']} (seed {d['seed']}, validation macro-F1 {d['val_f1']:.3f})"
    st.markdown(
        f"""
**Model.** {served or 'A fine-tuned transformer'}, trained on AfriSenti Swahili tweets
(Muhammad et al., 2023, CC BY 4.0). Model source: `{source}`.

**What it does.** It reads one tweet and returns one of three labels: positive, neutral or negative,
with a probability for each.

**Limits.**
- It was trained on tweets only; other kinds of text may be classified badly.
- It always picks one of the three labels, even for text that is not really about sentiment.
- Sentiment labels are subjective, and sarcasm and mixed Swahili/English are hard.
- The test set is small, so the scores in the results tab are only a rough guide.

**Privacy.** This page does not save what you type.
"""
    )
