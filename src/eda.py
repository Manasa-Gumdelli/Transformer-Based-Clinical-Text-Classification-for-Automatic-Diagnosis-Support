"""Exploratory data analysis outputs required by plan section 5.5.

Produces (in outputs/figures and outputs/tables):
  - class frequency table/chart before and after filtering
  - missing-value and duplicate summary (re-exported from the audit)
  - transcription word-count and token-length distributions
  - per-class median/IQR of document length
  - vocabulary size and most frequent terms after stop-word review
  - percentage of texts exceeding the transformer max sequence length
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from transformers import AutoTokenizer

from . import config
from .data import load_raw
from .preprocessing import clean_dataframe


def class_frequency_before_after():
    raw = load_raw()
    cleaned = clean_dataframe(raw)
    before = cleaned[config.LABEL_COLUMN].value_counts()

    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")
    after = processed[config.LABEL_COLUMN].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    before.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title(f"All specialties before filtering (n={len(before)})")
    axes[0].set_xlabel("Record count")
    axes[0].axvline(config.MIN_CLASS_COUNT, color="red", linestyle="--",
                     label=f">= {config.MIN_CLASS_COUNT} threshold")
    axes[0].legend()

    after.sort_values().plot.barh(ax=axes[1], color="seagreen")
    axes[1].set_title(f"Retained specialties after filtering (n={len(after)})")
    axes[1].set_xlabel("Record count")

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_frequency_before_after.png", dpi=150)
    plt.close(fig)

    before.rename("count").to_csv(config.TABLES_DIR / "class_frequency_before.csv")
    after.rename("count").to_csv(config.TABLES_DIR / "class_frequency_after.csv")
    return before, after


def document_length_analysis(processed: pd.DataFrame):
    processed = processed.copy()
    processed["word_count"] = processed[config.TEXT_COLUMN].str.split().apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(processed["word_count"], bins=50, color="slateblue")
    ax.set_title("Transcription word-count distribution (retained classes)")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Number of documents")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "word_count_distribution.png", dpi=150)
    plt.close(fig)

    per_class = processed.groupby(config.LABEL_COLUMN)["word_count"].agg(
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        mean="mean",
        n="count",
    )
    per_class["iqr"] = per_class["q3"] - per_class["q1"]
    per_class = per_class.sort_values("median", ascending=False)
    per_class.to_csv(config.TABLES_DIR / "per_class_document_length.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    order = per_class.index.tolist()
    processed[config.LABEL_COLUMN] = pd.Categorical(
        processed[config.LABEL_COLUMN], categories=order, ordered=True
    )
    processed.boxplot(column="word_count", by=config.LABEL_COLUMN, ax=ax, rot=90)
    ax.set_title("Word count by specialty")
    plt.suptitle("")
    ax.set_xlabel("")
    ax.set_ylabel("Word count")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "word_count_by_class_boxplot.png", dpi=150)
    plt.close(fig)

    return per_class


def token_length_analysis(processed: pd.DataFrame):
    local_tokenizer = config.TRANSFORMER_CHECKPOINT_DIR / "final_model"
    tokenizer_source = (
        local_tokenizer
        if (local_tokenizer / "tokenizer.json").exists()
        else config.TRANSFORMER_MODEL_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    texts = processed[config.TEXT_COLUMN].tolist()
    token_counts = [
        len(tokenizer.encode(t, add_special_tokens=True)) for t in texts
    ]
    processed = processed.copy()
    processed["n_tokens"] = token_counts

    pct_over_max = float((processed["n_tokens"] > config.MAX_SEQ_LENGTH).mean() * 100)
    pct_over_512 = float((processed["n_tokens"] > 512).mean() * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(processed["n_tokens"], bins=50, color="darkorange")
    ax.axvline(config.MAX_SEQ_LENGTH, color="red", linestyle="--",
               label=f"max_seq_length={config.MAX_SEQ_LENGTH}")
    ax.axvline(512, color="black", linestyle=":", label="BERT hard limit=512")
    ax.set_title("Bio_ClinicalBERT token-length distribution")
    ax.set_xlabel("Token count (incl. special tokens)")
    ax.set_ylabel("Number of documents")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "token_length_distribution.png", dpi=150)
    plt.close(fig)

    summary = {
        "tokenizer": config.TRANSFORMER_MODEL_NAME,
        "median_tokens": float(processed["n_tokens"].median()),
        "mean_tokens": float(processed["n_tokens"].mean()),
        "pct_exceeding_max_seq_length": round(pct_over_max, 2),
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "pct_exceeding_512_hard_limit": round(pct_over_512, 2),
    }
    with open(config.TABLES_DIR / "token_length_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def vocabulary_analysis(processed: pd.DataFrame, top_n=30):
    vectorizer = CountVectorizer(
        lowercase=True, stop_words="english", min_df=2, max_df=0.95
    )
    matrix = vectorizer.fit_transform(processed[config.TEXT_COLUMN])
    vocab_size = len(vectorizer.vocabulary_)

    term_freq = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    freq_table = (
        pd.DataFrame({"term": terms, "frequency": term_freq})
        .sort_values("frequency", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    freq_table.to_csv(config.TABLES_DIR / "top_terms.csv", index=False)

    summary = {
        "vocabulary_size_min_df2_max_df0.95": int(vocab_size),
        "n_sklearn_english_stopwords": len(ENGLISH_STOP_WORDS),
        "top_terms_path": "outputs/tables/top_terms.csv",
    }
    with open(config.TABLES_DIR / "vocabulary_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run():
    class_frequency_before_after()
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")
    per_class = document_length_analysis(processed)
    token_summary = token_length_analysis(processed)
    vocab_summary = vocabulary_analysis(processed)
    print("Per-class doc length (head):")
    print(per_class.head())
    print("Token length summary:", token_summary)
    print("Vocabulary summary:", vocab_summary)


if __name__ == "__main__":
    run()
