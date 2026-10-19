# Swahili Tweet Sentiment Classification

Classifying Swahili tweets as positive, neutral or negative with reference baselines, TF-IDF + Logistic Regression, a BiLSTM, and two fine-tuned African-language Transformers (AfriBERTa-base, AfroXLMR-base), plus a deployed web app.

| Resource | Link |
|---|---|
| Live app | https://summativetechniquei-x6jtcsn5dkzcft2gnskyjx.streamlit.app |
| Fine-tuned model (Hugging Face) | https://huggingface.co/453ivan/swahili-sentiment-afrisenti |
| Code and results (this repository) | https://github.com/Manzi453/Summative_Technique_I |
| Demo video | https://www.awesomescreenshot.com/video/57562961?key=2702f4389ab7cfd57daca6e03e6e940e |
| Final report | `report_and_script/Swahili_Sentiment_Report.docx` |

## 1. Problem

Swahili is widely spoken in East Africa, yet labelled data for sentiment analysis is limited compared with English. The goal is to build and compare classifiers for Swahili tweets, check whether the differences between models are larger than random variation, analyse the errors, and deploy a model that anyone can try.

## 2. Dataset

**AfriSenti** (Muhammad et al., 2023), Swahili part of SemEval-2023 Task 12, Subtask A. Tweets labelled positive, negative or neutral. Licence: CC BY 4.0. Source files: `github.com/afrisenti-semeval/afrisent-semeval-2023`.

- Official splits, used as they are: train 1,810, validation 453 (the official dev set), test 748. Total 3,011 tweets.
- Classes (train): neutral 1,072, positive 547, negative 191. Test: neutral 444, positive 224, negative 80.
- Cleaning: lower-case, remove links, collapse whitespace. The released tweets contain no hashtags, mentions or emoji.
- Tweets are short (mean 17 words). About 20% of validation and test word tokens never occur in training.
- Overlap between splits is negligible (3 validation and 1 test tweet also occur in training).

## 3. Methods

| Model | Description |
|---|---|
| Reference models | Always the most frequent class; random by class frequency. |
| TF-IDF + Logistic Regression | Word 1-2-gram, character 2-5-gram, or both; balanced class weights; C chosen on validation macro-F1. |
| BiLSTM | PyTorch, 100-d embeddings from scratch, 64 hidden units per direction, mean and max pooling, dropout 0.5, class-weighted loss; 5 seeds. |
| AfriBERTa-base, AfroXLMR-base | `castorini/afriberta_base`, `Davlan/afro-xlmr-base`; fine-tuned with AdamW, lr 3e-5, batch 16, up to 4 epochs, word embeddings frozen, early stopping on validation macro-F1, max length 128; 2 seeds each (shortened to fit an 8 GB laptop). |

Primary metric: macro-F1. Also accuracy, weighted F1 (the shared-task metric), macro precision and recall. Bootstrap 95% confidence intervals (1,000 resamples) and paired bootstrap differences on the test set.

The best validation run is saved to `models/best_sentiment` and served by the Streamlit app in `streamlit_app/` (PyTorch on the CPU).

## 4. Results

Test set (748 tweets), from `results/metrics_summary.csv`. Neural models: mean over seeds, standard deviation of macro-F1 in brackets.

| Model | Accuracy | Macro-F1 (std) | Weighted F1 |
|---|---|---|---|
| Majority class (always neutral) | 0.594 | 0.248 | 0.442 |
| Random by class frequency | 0.436 | 0.321 | 0.436 |
| TF-IDF + Logistic Regression | 0.580 | 0.483 | 0.574 |
| BiLSTM (5 seeds) | 0.475 | 0.410 (0.051) | 0.489 |
| AfriBERTa-base fine-tuned (2 seeds) | 0.608 | 0.558 (0.009) | 0.607 |
| AfroXLMR-base fine-tuned (2 seeds) | 0.577 | 0.562 (0.018) | 0.583 |

Both Transformers score higher than TF-IDF and the BiLSTM in a paired bootstrap on the test set (AfriBERTa +0.068, 95% interval [0.014, 0.119]; AfroXLMR +0.066, [0.013, 0.115] versus TF-IDF; all pairs in `results/metrics/bootstrap_paired_all.csv`), but the two Transformers cannot be told apart (+0.003, [-0.043, 0.045]). The gain is mostly in the negative class, and accuracy barely exceeds the 0.594 obtained by always predicting neutral. For context, the SemEval-2023 Swahili Task A results were weighted F1 63.40 (the AfroXLMR-large baseline) and 65.68 (best system); our setups differ (base-size models, frozen embeddings, at most 4 epochs, 2 seeds), so this is context and not a direct comparison. The report has the full analysis.

## 5. Repository structure

```
notebooks/            01 to 04, run in this order (VS Code, using the project's .venv)
data/processed/       train.csv, val.csv, test.csv
figures/              EDA, training-curve figures
results/              metrics_summary.csv, confusion matrices, errors, predictions, bootstrap tables
streamlit_app/        the Streamlit web app
scripts/              extra_analysis.py: bootstrap for all model pairs, per-class scores, unseen-vocabulary analysis
report_and_script/    the report and the demo-video script
```

## 6. How to reproduce

1. `01_data_exploration.ipynb` downloads AfriSenti Swahili, explores it and writes the splits.
2. `02_baseline_tfidf_logreg.ipynb` reference models and TF-IDF baseline.
3. `03_bilstm.ipynb` BiLSTM, five seeds (a few minutes on the CPU).
4. `04_finetune_african_models.ipynb` fine-tunes the two Transformers (a GPU is recommended; about an hour on an 8 GB M2 Mac with the settings below), bootstrap comparison, optionally uploads the best model to the Hugging Face Hub.
5. `python scripts/extra_analysis.py` computes, from the saved predictions, the analyses used in the report.

The reported results used `SEEDS = [1, 2]` and `EPOCHS = 4` in notebook 4. The notebook may currently be set to a quick mode (1 seed, 2 epochs); running it overwrites `results/` with weaker numbers, so set the reported values first to reproduce the report.

Notebook 4 asks for a Hugging Face token with write access at run time, only to upload the model.

Run the app locally:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

The app loads `models/best_sentiment` (saved by notebook 4). If that folder is missing it downloads the model from the Hugging Face repo recorded in `results/metrics/deployed_model.json`. Set the `MODEL_ID` environment variable to use another repo or a local folder.

## 7. Limitations

- Small test set (748 tweets, 80 negative): per-class scores are noisy.
- One setting per neural model; little hyper-parameter tuning; fine-tuning used 2 seeds, at most 4 epochs and frozen word embeddings.
- Sentiment labels are subjective; sarcasm and code-switching are hard.
- Pretraining data of the two Transformers may overlap with public tweets; this could not be checked.
- The app always returns one of three labels and is trained on tweets only.
- Free hosting: the app sleeps when idle and has no custom domain.

## 8. Existing resources and own work

Existing resources: the AfriSenti dataset, the pretrained models `castorini/afriberta_base` and `Davlan/afro-xlmr-base`, and the libraries scikit-learn, PyTorch, Hugging Face Transformers and Tokenizers, and Streamlit.

Implemented here: data checks and cleaning, reference and TF-IDF baselines, the BiLSTM, the multi-seed fine-tuning pipeline, bootstrap evaluation, error analysis, and the Streamlit web app.


## 9. References

- Muhammad, S. H., Abdulmumin, I., Ayele, A. A., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.862
- Muhammad, S. H., Abdulmumin, I., Yimam, S. M., et al. (2023). SemEval-2023 Task 12: Sentiment analysis for African languages (AfriSenti-SemEval). arXiv:2304.06845.
- Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. https://aclanthology.org/2021.mrl-1.11
- Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*. https://aclanthology.org/2022.coling-1.382

The full reference list is in the report.
