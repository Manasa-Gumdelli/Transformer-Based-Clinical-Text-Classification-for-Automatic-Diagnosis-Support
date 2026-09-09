# Clinical Text Classification (Medical-Specialty Prediction)

Research prototype comparing traditional NLP baselines (TF-IDF + Multinomial
Naive Bayes, TF-IDF + Linear SVM) against a domain-adapted transformer
(Bio_ClinicalBERT) on the Kaggle Medical Transcriptions dataset
(`mtsamples.csv`), following `Clinical_Text_Classification_Implementation_Plan.docx`.

**This is a research prototype. It predicts the medical specialty
associated with a transcription's document style and content — it does not
diagnose disease, and its output must not be used clinically without expert
human review.** See [Ethics and scope](#ethics-and-scope) below.

## Task

- Input: `transcription` free text.
- Target: `medical_specialty`, restricted to the 12 specialties with ≥100
  records in the raw dataset. After removing blank/missing transcriptions and
  collapsing exact duplicate transcriptions with conflicting labels, the primary
  dataset contains 2,236 unique transcription examples.
- Problem type: single-label multiclass classification.
- Primary metric: macro-F1 (equal weight per class, robust to class imbalance).

## Project structure

```
clinical-text-classification/
├── data/
│   ├── raw/mtsamples.csv          # gitignored — place the Kaggle CSV here
│   ├── processed/                 # cleaned dataset, label map, manifest
│   └── splits/                    # fixed train/val/test record-id lists
├── src/
│   ├── config.py                  # paths, seeds, label scope, model hyperparameters
│   ├── data.py                    # loading + dataset audit
│   ├── preprocessing.py           # cleaning, class-scope filtering, leakage-safe split
│   ├── eda.py                     # exploratory analysis figures/tables
│   ├── train_baselines.py         # TF-IDF + Naive Bayes / Linear SVM
│   ├── train_transformer.py       # Bio_ClinicalBERT fine-tuning
│   ├── evaluate.py                # single held-out test evaluation + comparison
│   └── explain.py                 # feature/error/confidence analysis
├── models/                        # gitignored — baseline pipelines + transformer checkpoint
├── outputs/
│   ├── figures/                   # all plots (PNG)
│   └── tables/                    # all tables/metrics (CSV/JSON)
├── tests/
├── requirements.txt
└── configs/
```

## Setup

Requires Python 3.11 or 3.12 (Python ≥3.14 is not recommended — several
PyTorch/Transformers dependencies are unverified on it at time of writing).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle dataset at `data/raw/mtsamples.csv` (already gitignored).

## Reproducing the full pipeline

Run each stage from the project root, in order. Every stage is deterministic
given `RANDOM_SEED = 42` (see `src/config.py`) and writes its outputs to
`outputs/tables/`, `outputs/figures/`, `data/processed/`, `data/splits/` or
`models/`.

```bash
python3 -m src.data              # dataset audit -> outputs/tables/dataset_audit_summary.json
python3 -m src.preprocessing     # cleaning, 12-class scope, fixed 70/15/15 split, manifest
python3 -m src.eda                # EDA figures/tables, incl. token-length audit
python3 -m src.train_baselines   # NB + Linear SVM, 5-fold CV grid search on train only
python3 -m src.train_transformer # Bio_ClinicalBERT fine-tune, early stopping on val macro-F1
python3 -m src.evaluate          # ONE held-out test evaluation of all three models
python3 -m src.explain           # SVM feature weights, confusion pairs, error taxonomy
```

`evaluate.py` should only be run after all baseline/transformer hyperparameter
choices are frozen — this mirrors the plan's requirement that the test set is
touched exactly once.

## Key methodological decisions (and why)

- **Leakage-aware split.** 83% of retained transcriptions (3,231 of 3,894
  rows) share their exact text with at least one other row (many are
  templated/boilerplate notes, e.g. redacted newborn-exam forms, that recur
  across different specialty labels).
  Because those identical inputs often have conflicting labels, the supervised
  target is not identifiable from transcription-only text at the row level.
  `preprocessing.py` now collapses each exact whitespace-normalised duplicate
  transcription group to one deterministic majority-labeled record before the
  grouped stratified split. The audit is saved to
  `outputs/tables/duplicate_resolution_summary.json`.
- **Excluded features.** `keywords`, `sample_name` and `description` are
  dropped from the model input because they can leak the specialty label
  almost directly (plan section 5.2).
- **Sequence length.** The token-length audit
  (`outputs/tables/token_length_summary.json`) showed 89.36% of transcriptions
  exceed 256 tokens and 63.46% exceed BERT's 512-token hard limit. The
  transformer therefore splits long documents into overlapping 512-token
  windows with stride 128, trains on document chunks, and aggregates chunk
  logits back to one document-level prediction for validation and test
  evaluation. This implements the plan's chunk-aggregation extension (S3)
  without adding the leaky metadata fields.
- **Some "specialties" are document types, not clinical fields**
  (`Consult - History and Phy.`, `SOAP / Chart / Progress Notes`,
  `Discharge Summary`, `General Medicine`), so their text can resemble any
  true specialty. This is a genuine, data-driven source of difficulty and is
  called out explicitly in the error taxonomy rather than treated as a
  modelling bug.
- **SVM confidence scores.** `LinearSVC` has no `predict_proba`; evaluation
  converts the one-vs-rest decision scores with a softmax transform so a
  comparable confidence score is available for the confidence-distribution
  analysis in `explain.py`.
- **Device.** Training/inference use Apple MPS (`torch.backends.mps`) when
  available, else CPU; mixed precision (`fp16`) is enabled only on CUDA
  because MPS autocast is not reliably supported by this Trainer version.

## Current results

After duplicate-label cleanup and a full rerun with chunk-aggregated transformer
inference, the held-out test set contains 320 examples. Bio_ClinicalBERT now has
the best macro-F1 and ties the Linear SVM on accuracy:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Naive Bayes | 0.806 | 0.630 | 0.802 |
| Linear SVM | 0.838 | 0.692 | 0.832 |
| Bio_ClinicalBERT | 0.838 | 0.707 | 0.835 |

Both the SVM and chunk-aggregated transformer reach the expected ~80% accuracy
target without adding the leaky `keywords`, `sample_name` or `description`
fields.

## Ethics and scope

- Dataset: Kaggle "Medical Transcriptions" (`mtsamples.csv`) — publicly
  released, de-identified sample transcriptions. No re-identification is
  attempted or possible from this data.
- The raw dataset, processed CSVs, split indices, model checkpoints and any
  file that could embed transcription text are excluded from version control
  (`.gitignore`). Only aggregate statistics, figures and short synthetic/
  paraphrased illustrations belong in a dissertation or public repository —
  never verbatim transcription text (plan section 9/10). The one intentional
  exception is `svm_top_features_per_class.csv`, which lists individual
  vocabulary terms (words/character n-grams) by TF-IDF weight — these are
  common terms, not spans of any single patient's record.
- This prototype predicts document/specialty routing, not a clinical
  diagnosis. It must not be used to inform real patient care, and results
  should be interpreted with the class-imbalance, label-noise and
  document-type caveats documented in `outputs/tables/`.

## Status against the plan's acceptance criteria

See `outputs/tables/` for the generated evidence backing each item in plan
section 14.1 (dataset audit, fixed splits, per-class metrics, confusion
matrices, error taxonomy, compute/memory, model-comparison table).
