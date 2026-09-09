"""Cleaning, class-scope selection, duplicate-label cleanup and fixed splits.

Implements plan sections 2.1, 2.2, 5.1, 5.2 and 5.4:
  - retain only specialties with >= MIN_CLASS_COUNT records
  - drop rows with missing/blank transcription
  - exclude leaky columns (keywords, sample_name, description) from model input
  - collapse exact-duplicate transcriptions with conflicting labels
  - group any remaining duplicate transcriptions so they cannot cross a split boundary
  - fixed stratified(-group) 70/15/15 train/val/test split, random_state=42
"""

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data import load_raw


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns]).copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(str).str.strip()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).where(
        df[config.TEXT_COLUMN].notna(), other=np.nan
    )
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    return df


def select_scope(df: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, list[str]]:
    """Retain classes with >= min_count records (computed before dropping missing
    transcriptions, per plan 2.1), then drop rows with missing/blank transcription."""
    label_counts = df[config.LABEL_COLUMN].value_counts()
    retained_labels = sorted(label_counts[label_counts >= min_count].index.tolist())
    scoped = df[df[config.LABEL_COLUMN].isin(retained_labels)].copy()
    is_missing = scoped[config.TEXT_COLUMN].isna() | (
        scoped[config.TEXT_COLUMN].fillna("").str.strip() == ""
    )
    scoped = scoped.loc[~is_missing].reset_index(drop=True)
    return scoped, retained_labels


def build_label_mapping(labels: list[str]) -> dict:
    return {label: idx for idx, label in enumerate(sorted(labels))}


def normalise_transcription(text: pd.Series) -> pd.Series:
    return text.str.lower().str.split().str.join(" ")


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per exact (whitespace-normalised) transcription text, so identical
    transcriptions cannot be split across train/val/test (plan 5.2/5.4)."""
    group_ids, _ = pd.factorize(normalise_transcription(df[config.TEXT_COLUMN]))
    return pd.Series(group_ids, index=df.index, name="group_id")


def collapse_duplicate_transcriptions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse exact duplicate transcription text to one majority-labeled record.

    In mtsamples, many identical transcriptions are repeated under different
    specialty labels. With transcription-only input, those rows are not
    learnable as separate examples. This deterministic collapse treats them as
    label-noise duplicates rather than allowing impossible contradictory targets.
    """
    scoped = df.copy()
    scoped["_normalised_transcription"] = normalise_transcription(scoped[config.TEXT_COLUMN])
    rows = []
    duplicate_groups = 0
    conflicting_groups = 0
    duplicate_rows_removed = 0
    conflicting_rows_collapsed = 0

    for _, group in scoped.groupby("_normalised_transcription", sort=False):
        label_counts = group[config.LABEL_COLUMN].value_counts()
        duplicate_groups += int(len(group) > 1)
        duplicate_rows_removed += max(0, len(group) - 1)
        if len(label_counts) > 1:
            conflicting_groups += 1
            conflicting_rows_collapsed += len(group)

        majority_count = label_counts.max()
        tied_labels = label_counts[label_counts == majority_count].index.tolist()
        majority_label = next(
            label for label in group[config.LABEL_COLUMN].tolist()
            if label in tied_labels
        )

        majority_rows = group[group[config.LABEL_COLUMN] == majority_label]
        representative = majority_rows.iloc[0].copy()
        representative[config.LABEL_COLUMN] = majority_label
        rows.append(representative.drop(labels=["_normalised_transcription"]))

    collapsed = pd.DataFrame(rows).reset_index(drop=True)
    summary = {
        "enabled": True,
        "strategy": (
            "Exact whitespace-normalised duplicate transcriptions are collapsed "
            "to one deterministic majority-labeled record; label ties use the "
            "first source-file occurrence among tied labels."
        ),
        "rows_before_collapse": int(len(scoped)),
        "rows_after_collapse": int(len(collapsed)),
        "duplicate_text_groups": int(duplicate_groups),
        "conflicting_label_duplicate_groups": int(conflicting_groups),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "rows_in_conflicting_duplicate_groups": int(conflicting_rows_collapsed),
    }
    return collapsed, summary


def grouped_stratified_split(df: pd.DataFrame, label_ids: pd.Series, group_ids: pd.Series,
                              train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC,
                              test_frac=config.TEST_FRAC, seed=config.RANDOM_SEED) -> pd.Series:
    """Two-stage StratifiedGroupKFold approximating a fixed 70/15/15 split while
    guaranteeing that no group (exact-duplicate transcription) crosses a split."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    n_splits_stage1 = round(1 / test_frac)
    sgkf1 = StratifiedGroupKFold(n_splits=n_splits_stage1, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf1.split(df, label_ids, group_ids))

    remaining_frac = 1 - test_frac
    val_of_remaining = val_frac / remaining_frac
    n_splits_stage2 = round(1 / val_of_remaining)
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_stage2, shuffle=True, random_state=seed)
    sub_label_ids = label_ids.iloc[trainval_idx].reset_index(drop=True)
    sub_group_ids = group_ids.iloc[trainval_idx].reset_index(drop=True)
    sub_train_pos, sub_val_pos = next(sgkf2.split(
        np.zeros(len(trainval_idx)), sub_label_ids, sub_group_ids
    ))
    train_idx = trainval_idx[sub_train_pos]
    val_idx = trainval_idx[sub_val_pos]

    split = pd.Series(index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split


def verify_no_group_leakage(df: pd.DataFrame) -> None:
    leaking = df.groupby("group_id")["split"].nunique()
    n_leaking_groups = int((leaking > 1).sum())
    if n_leaking_groups:
        raise AssertionError(
            f"{n_leaking_groups} duplicate-text groups were split across partitions"
        )


def run() -> pd.DataFrame:
    raw = load_raw()
    cleaned = clean_dataframe(raw)
    scoped, retained_labels = select_scope(cleaned, config.MIN_CLASS_COUNT)
    rows_after_scope = len(scoped)

    if config.COLLAPSE_DUPLICATE_TRANSCRIPTIONS:
        scoped, duplicate_summary = collapse_duplicate_transcriptions(scoped)
    else:
        duplicate_summary = {
            "enabled": False,
            "rows_before_collapse": int(rows_after_scope),
            "rows_after_collapse": int(len(scoped)),
        }

    label_map = build_label_mapping(retained_labels)
    scoped["label_id"] = scoped[config.LABEL_COLUMN].map(label_map)
    scoped["group_id"] = assign_groups(scoped)
    scoped["record_id"] = scoped.index

    scoped["split"] = grouped_stratified_split(
        scoped, scoped["label_id"], scoped["group_id"]
    )
    verify_no_group_leakage(scoped)

    model_columns = ["record_id", config.TEXT_COLUMN, config.LABEL_COLUMN,
                      "label_id", "group_id", "split"]
    processed = scoped[model_columns].reset_index(drop=True)
    processed.to_csv(config.PROCESSED_DIR / "dataset.csv", index=False)

    for split_name in ("train", "val", "test"):
        subset = processed.loc[processed["split"] == split_name, "record_id"]
        subset.to_csv(config.SPLITS_DIR / f"{split_name}_ids.csv", index=False)

    with open(config.PROCESSED_DIR / "label_mapping.json", "w") as f:
        json.dump(label_map, f, indent=2)

    split_counts = processed["split"].value_counts().to_dict()
    class_split_table = (
        processed.groupby([config.LABEL_COLUMN, "split"]).size().unstack(fill_value=0)
    )
    class_split_table = class_split_table.reindex(
        columns=["train", "val", "test"], fill_value=0
    )
    class_split_table["total"] = class_split_table.sum(axis=1)
    class_split_table = class_split_table.sort_values("total", ascending=False)
    class_split_table.to_csv(config.TABLES_DIR / "class_split_table.csv")

    with open(config.TABLES_DIR / "duplicate_resolution_summary.json", "w") as f:
        json.dump(duplicate_summary, f, indent=2)

    manifest = {
        "source": config.DATASET_SOURCE,
        "retrieval_date": str(date.today()),
        "licence_note": config.DATASET_LICENCE_NOTE,
        "raw_row_count": int(len(raw)),
        "n_classes_full": int(cleaned[config.LABEL_COLUMN].nunique()),
        "min_class_count_threshold": config.MIN_CLASS_COUNT,
        "n_retained_classes": len(retained_labels),
        "retained_classes": retained_labels,
        "n_rows_after_scope_and_missing_removal_before_duplicate_collapse": int(rows_after_scope),
        "n_rows_after_scope_and_missing_removal": int(len(processed)),
        "duplicate_resolution": duplicate_summary,
        "n_duplicate_text_groups": int(processed["group_id"].nunique()) - int(
            (processed["group_id"].value_counts() == 1).sum()
        ),
        "excluded_leaky_columns": config.LEAKY_COLUMNS,
        "split_strategy": (
            f"Grouped stratified {config.TRAIN_FRAC}/{config.VAL_FRAC}/{config.TEST_FRAC} "
            f"split via two-stage StratifiedGroupKFold, random_state={config.RANDOM_SEED}. "
            "Groups = exact-duplicate (whitespace-normalised, lower-cased) transcription "
            "text. Duplicate transcriptions are collapsed before splitting when "
            "COLLAPSE_DUPLICATE_TRANSCRIPTIONS=True, so contradictory duplicate-label "
            "rows cannot create impossible transcription-only targets."
        ),
        "split_row_counts": split_counts,
        "label_mapping_path": "data/processed/label_mapping.json",
    }
    with open(config.PROCESSED_DIR / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return processed


if __name__ == "__main__":
    result = run()
    print(f"Processed {len(result)} rows across {result[config.LABEL_COLUMN].nunique()} classes")
    print(result["split"].value_counts())
