"""Dataset loading and schema/quality validation (plan section 2 and 5.1)."""

import json
from datetime import date

import pandas as pd

from . import config

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "description",
    "medical_specialty",
    "sample_name",
    "transcription",
    "keywords",
]


def load_raw(path=None) -> pd.DataFrame:
    """Load mtsamples.csv and confirm schema/encoding/row count."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path, encoding="utf-8")
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Unexpected schema, missing columns: {missing_cols}")
    return df


def audit(df: pd.DataFrame) -> dict:
    """Produce the dataset audit summary described in plan sections 2 and 5.1."""
    n_rows = len(df)
    is_missing_transcription = df[config.TEXT_COLUMN].isna() | (
        df[config.TEXT_COLUMN].fillna("").str.strip() == ""
    )
    missing_transcription = int(is_missing_transcription.sum())
    missing_keywords = df["keywords"].isna().sum()
    n_labels = df[config.LABEL_COLUMN].astype(str).str.strip().nunique()

    non_blank_transcriptions = df.loc[~is_missing_transcription, config.TEXT_COLUMN].str.strip()
    exact_dup_transcriptions = int(non_blank_transcriptions.duplicated(keep=False).sum())
    exact_dup_rows = int(df.duplicated(keep=False).sum())

    label_counts = (
        df[config.LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("medical_specialty")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    summary = {
        "retrieved_date": str(date.today()),
        "source": config.DATASET_SOURCE,
        "licence_note": config.DATASET_LICENCE_NOTE,
        "n_rows": int(n_rows),
        "n_columns": int(df.shape[1]),
        "n_distinct_labels": int(n_labels),
        "n_missing_transcription": int(missing_transcription),
        "n_missing_keywords": int(missing_keywords),
        "n_duplicate_transcriptions_involved": exact_dup_transcriptions,
        "n_fully_duplicate_rows": exact_dup_rows,
        "n_classes_ge_100": int((label_counts["count"] >= config.MIN_CLASS_COUNT).sum()),
        "n_classes_ge_50": int((label_counts["count"] >= config.SENSITIVITY_MIN_CLASS_COUNT).sum()),
    }
    return summary, label_counts


def save_audit(summary: dict, label_counts: pd.DataFrame) -> None:
    with open(config.TABLES_DIR / "dataset_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    label_counts.to_csv(config.TABLES_DIR / "label_counts_full.csv", index=False)


if __name__ == "__main__":
    raw_df = load_raw()
    audit_summary, counts = audit(raw_df)
    save_audit(audit_summary, counts)
    print(json.dumps(audit_summary, indent=2))
