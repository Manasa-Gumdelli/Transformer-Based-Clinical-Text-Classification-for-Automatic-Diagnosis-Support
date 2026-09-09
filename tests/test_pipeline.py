"""Lightweight checks that the processed dataset and splits satisfy the plan's
leakage/scope requirements. Run with: pytest tests/ (after src.preprocessing)."""

import json

import pandas as pd
import pytest

from src import config

DATASET_PATH = config.PROCESSED_DIR / "dataset.csv"


@pytest.fixture(scope="module")
def processed():
    if not DATASET_PATH.exists():
        pytest.skip("Run `python -m src.preprocessing` before the test suite.")
    return pd.read_csv(DATASET_PATH)


def test_only_scoped_classes_present(processed):
    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    assert set(processed[config.LABEL_COLUMN].unique()) == set(label_map.keys())
    assert len(label_map) == 12


def test_no_missing_transcriptions(processed):
    assert processed[config.TEXT_COLUMN].isna().sum() == 0
    assert (processed[config.TEXT_COLUMN].str.strip() == "").sum() == 0


def test_splits_partition_dataset(processed):
    assert set(processed["split"].unique()) == {"train", "val", "test"}
    assert processed["record_id"].is_unique


def test_no_duplicate_text_crosses_split_boundary(processed):
    leaking = processed.groupby("group_id")["split"].nunique()
    assert (leaking > 1).sum() == 0


def test_duplicate_transcriptions_collapsed(processed):
    normalised = processed[config.TEXT_COLUMN].str.lower().str.split().str.join(" ")
    assert normalised.is_unique


def test_split_proportions_close_to_target(processed):
    fracs = processed["split"].value_counts(normalize=True)
    assert abs(fracs["train"] - config.TRAIN_FRAC) < 0.05
    assert abs(fracs["val"] - config.VAL_FRAC) < 0.05
    assert abs(fracs["test"] - config.TEST_FRAC) < 0.05
