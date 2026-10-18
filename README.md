# Swahili Tweet Sentiment Classification

Classifying Swahili tweets as positive, neutral or negative with reference baselines, TF-IDF + Logistic Regression, a BiLSTM, and two fine-tuned African-language Transformers (AfriBERTa-base, AfroXLMR-base), plus a deployed web app.

| Resource | Link |
|---|---|
| Live app | FILL: your .streamlit.app link |
| Fine-tuned model (Hugging Face) | FILL: https://huggingface.co/YOUR-USERNAME/YOUR-MODEL-NAME |
| Code and results (this repository) | FILL: https://github.com/YOUR-USERNAME/swahili-sentiment-afrisenti |
| Demo video | FILL: link after upload |
| Final report | FILL: link or file name |

> Drafting note: delete every line that starts with FILL, and the result numbers below must come from `results/metrics_summary.csv` after you run the notebooks. Delete this note when done.

## 1. Problem

Swahili is widely spoken in East Africa, yet labelled data for sentiment analysis is limited compared with English. The goal is to build and compare classifiers for Swahili tweets, check whether the differences between models are larger than random variation, analyse the errors, and deploy a model that anyone can try.

## 2. Dataset

**AfriSenti** (Muhammad et al., 2023), Swahili part of SemEval-2023 Task 12, Subtask A. Tweets labelled positive, negative or neutral. Licence: CC BY 4.0. Source files: `github.com/afrisenti-semeval/afrisent-semeval-2023`.

- Official splits, used as they are: train 1,810, validation 453 (the official dev set), test 748. Total 3,011 tweets.
- Classes (train): neutral 1,072, positive 547, negative 191. Test: neutral 444, positive 224, negative 80.
- Cleaning: lower-case, remove links, collapse whitespace. Hashtags, mentions and emoji are kept.
- Tweets are short (mean 17 words). About 20% of validation and test word tokens never occur in training.
- Overlap between splits is negligible (3 validation and 1 test tweet also occur in training).

## 3. Methods

| Model | Description |
|---|---|
| Reference models | Always the most frequent class; random by class frequency. |
| TF-IDF + Logistic Regression | Word 1-2-gram, character 2-5-gram, or both; balanced class weights; C chosen on validation macro-F1. |
| BiLSTM | PyTorch, 100-d embeddings from scratch, 64 hidden units per direction, mean and max pooling, dropout 0.5, class-weighted loss; 5 seeds. |
| AfriBERTa-base, AfroXLMR-base | `castorini/afriberta_base`, `Davlan/afro-xlmr-base`; fine-tuned with AdamW, lr 3e-5, batch 16, up to 8 epochs, early stopping on validation macro-F1, max length 128; 5 seeds each. |

Primary metric: macro-F1. Also accuracy, weighted F1 (the shared-task metric), macro precision and recall. Bootstrap 95% confidence intervals (1,000 resamples) and paired bootstrap differences on the test set.

The best validation run is saved to `models/best_sentiment` and served by the Streamlit app in `streamlit_app/` (PyTorch on the CPU).

## 4. Results

FILL: paste the table from `results/metrics_summary.csv` (accuracy, macro-F1 with standard deviation over seeds, weighted F1) and 2 to 3 sentences on the main findings, including whether differences are larger than the bootstrap intervals (`results/metrics/bootstrap_paired_diff.csv`). For context, the SemEval-2023 Swahili Task A results were weighted F1 63.40 (organisers' AfroXLMR-large baseline) and 65.68 (best system); setups differ, so this is context and not a direct comparison.

## 5. Repository structure

```
notebooks/            01 to 04, run in this order (VS Code, using the project's .venv)
data/processed/       train.csv, val.csv, test.csv
figures/              EDA, training-curve figures
results/              metrics_summary.csv, confusion matrices, errors, predictions, bootstrap tables
streamlit_app/        the Streamlit web app
```

## 6. How to reproduce

1. `01_data_exploration.ipynb` downloads AfriSenti Swahili, explores it and writes the splits.
2. `02_baseline_tfidf_logreg.ipynb` reference models and TF-IDF baseline.
3. `03_bilstm.ipynb` BiLSTM, five seeds (GPU runtime).
4. `04_finetune_african_models.ipynb` fine-tunes the two Transformers (GPU runtime), bootstrap comparison, uploads the best model to the Hugging Face Hub.

Notebook 4 asks for a Hugging Face token with write access at run time, only to upload the model.

Run the app locally:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

The app loads `models/best_sentiment` (saved by notebook 4). If that folder is missing it downloads the model from the Hugging Face repo recorded in `results/metrics/deployed_model.json`. Set the `MODEL_ID` environment variable to use another repo or a local folder.

## 7. Limitations

- Small test set (748 tweets, 80 negative): per-class scores are noisy.
- One setting per neural model; little hyper-parameter tuning.
- Sentiment labels are subjective; sarcasm and code-switching are hard.
- Pretraining data of the two Transformers may overlap with public tweets; this could not be checked.
- The app always returns one of three labels and is trained on tweets only.
- Free hosting: the app sleeps when idle and has no custom domain.

## 8. Existing resources and own work

Existing resources: the AfriSenti dataset, the pretrained models `castorini/afriberta_base` and `Davlan/afro-xlmr-base`, and the libraries scikit-learn, PyTorch, Hugging Face Transformers and Tokenizers, and Streamlit.

Implemented here: data checks and cleaning, reference and TF-IDF baselines, the BiLSTM, the multi-seed fine-tuning pipeline, bootstrap evaluation, error analysis, and the Streamlit web app.

## 9. Use of AI tools

Claude (Anthropic) was used as an assistant for writing and debugging notebook and app code and for drafting this README and the report. All code was run by the project author in her own Colab sessions, and results in this repository come from those runs.

## 10. References

- Muhammad, S. H., Abdulmumin, I., Ayele, A. A., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.862
- Muhammad, S. H., Abdulmumin, I., Yimam, S. M., et al. (2023). SemEval-2023 Task 12: Sentiment analysis for African languages (AfriSenti-SemEval). arXiv:2304.06845.
- Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. https://aclanthology.org/2021.mrl-1.11
- Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*. https://aclanthology.org/2022.coling-1.382

The full reference list is in the report.
