"""Extra analyses for the report, computed from the saved predictions (no model is run).

Run from the project folder after notebooks 1 to 4:  python scripts/extra_analysis.py

Writes to results/metrics/:
  per_class_f1.csv          precision, recall and F1 per class for each model's best-validation run
  bootstrap_paired_all.csv  paired bootstrap difference in macro-F1 for every pair of models
  unseen_vocab_analysis.csv accuracy by how many words in a tweet never occur in the training set
  error_overlap.csv         how many models get each test tweet wrong, and errors that involve "neutral"
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "metrics"
LABELS = ["negative", "neutral", "positive"]
FILES = {"TF-IDF + LogReg": "tfidf", "BiLSTM": "bilstm", "AfriBERTa-base": "afriberta_base", "AfroXLMR-base": "afroxlmr_base"}

train = pd.read_csv(ROOT / "data/processed/train.csv")
test = pd.read_csv(ROOT / "data/processed/test.csv")
y = test["label_id"].values
pred = {}
for name, f in FILES.items():
    d = pd.read_csv(ROOT / f"results/predictions/{f}_test_preds.csv")
    assert (d["true"].values == y).all(), "test order differs"
    pred[name] = d["pred"].values

# 1. per-class scores
rows = []
for name, p in pred.items():
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    for i, lab in enumerate(LABELS):
        rows.append({"model": name, "class": lab, "precision": round(pr[i], 4), "recall": round(rc[i], 4),
                     "f1": round(f1[i], 4), "support": int(sup[i])})
pd.DataFrame(rows).to_csv(OUT / "per_class_f1.csv", index=False)

# 2. paired bootstrap for every pair (same resampling as notebook 4: seed 0, 1,000 resamples of the test set)
rng = np.random.default_rng(0)
B = 1000
idx = rng.integers(0, len(y), size=(B, len(y)))
boot = {n: np.array([f1_score(y[i], p[i], average="macro") for i in idx]) for n, p in pred.items()}
rows = []
for a, b in combinations(pred, 2):
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append({"model_a": a, "model_b": b,
                 "macro_f1_a": round(f1_score(y, pred[a], average="macro"), 4),
                 "macro_f1_b": round(f1_score(y, pred[b], average="macro"), 4),
                 "mean_diff_a_minus_b": round(d.mean(), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                 "interval_excludes_zero": bool(lo > 0 or hi < 0)})
pd.DataFrame(rows).to_csv(OUT / "bootstrap_paired_all.csv", index=False)

# 3. unseen vocabulary: share of a tweet's words that never occur in the training set
vocab = {w for t in train["text_clean"].fillna("") for w in t.split()}
words = [t.split() for t in test["text_clean"].fillna("")]
share = np.array([sum(w not in vocab for w in ws) / max(len(ws), 1) for ws in words])
groups = {"low (<=10% unseen words)": share <= 0.10, "middle (10-30%)": (share > 0.10) & (share < 0.30), "high (>=30%)": share >= 0.30}
rng = np.random.default_rng(0)
rows = []
for gname, mask in groups.items():
    r = {"group": gname, "n_tweets": int(mask.sum()), "n_negative": int((y[mask] == 0).sum())}
    for n, p in pred.items():
        r[f"acc_{n}"] = round(accuracy_score(y[mask], p[mask]), 4)
    rows.append(r)


def gap_ci(model, mask, base="TF-IDF + LogReg", B=1000):
    ids = np.where(mask)[0]
    g = []
    for _ in range(B):
        s = rng.choice(ids, len(ids))
        g.append(accuracy_score(y[s], pred[model][s]) - accuracy_score(y[s], pred[base][s]))
    return np.array(g)


for model in ["AfriBERTa-base", "AfroXLMR-base"]:
    g_low, g_high = gap_ci(model, groups["low (<=10% unseen words)"]), gap_ci(model, groups["high (>=30%)"])
    diff = g_high - g_low
    rows.append({"group": f"{model} minus TF-IDF accuracy: low group {g_low.mean():+.3f} [{np.percentile(g_low, 2.5):+.3f}, {np.percentile(g_low, 97.5):+.3f}]; "
                          f"high group {g_high.mean():+.3f} [{np.percentile(g_high, 2.5):+.3f}, {np.percentile(g_high, 97.5):+.3f}]; "
                          f"high minus low {diff.mean():+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]"})
pd.DataFrame(rows).to_csv(OUT / "unseen_vocab_analysis.csv", index=False)
rows_overview = {"share_of_tweets_with_any_unseen_word": round(float((share > 0).mean()), 4)}

# 4. error overlap and neutral involvement
wrong = np.array([p != y for p in pred.values()])
n_wrong = wrong.sum(0)
rows = [{"item": f"test tweets wrong in {k} of 4 models", "count": int((n_wrong == k).sum())} for k in range(5)]
for name, p in pred.items():
    err = p != y
    neutral_involved = int(((y[err] == 1) | (p[err] == 1)).sum())
    rows.append({"item": f"{name}: errors that involve the neutral class", "count": neutral_involved, "of_total_errors": int(err.sum())})
pd.DataFrame(rows).to_csv(OUT / "error_overlap.csv", index=False)
print("wrote", ", ".join(["per_class_f1.csv", "bootstrap_paired_all.csv", "unseen_vocab_analysis.csv", "error_overlap.csv"]), "| tweets with any unseen word:", rows_overview)
