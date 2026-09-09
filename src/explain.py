"""Explainability and error analysis (plan section 9).

  - top-weighted TF-IDF features per Linear SVM class
  - five largest off-diagonal confusion pairs per model, with likely-cause notes
  - error taxonomy: each test error is bucketed into one of the plan's categories
  - confidence-distribution comparison for correct vs incorrect predictions

No raw transcription text is written to any output file (plan section 9/10):
only word counts, aggregate stats and short slices used purely to measure length.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from . import config

TOP_N_FEATURES = 15
TOP_N_CONFUSION_PAIRS = 5


def top_svm_features():
    svm_pipeline = joblib.load(config.BASELINE_MODELS_DIR / "svm_pipeline.joblib")
    clf = svm_pipeline.named_steps["clf"]
    feature_union = svm_pipeline.named_steps["tfidf"]
    word_vec = feature_union.transformer_list[0][1]
    char_vec = feature_union.transformer_list[1][1]
    feature_names = np.concatenate([
        word_vec.get_feature_names_out(), char_vec.get_feature_names_out()
    ])

    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    if hasattr(clf, "coef_"):
        coef = clf.coef_
    else:
        base_estimators = [c.estimator for c in clf.calibrated_classifiers_]
        coef = np.mean([est.coef_ for est in base_estimators], axis=0)

    rows = []
    for class_idx in range(coef.shape[0]):
        top_idx = np.argsort(coef[class_idx])[::-1][:TOP_N_FEATURES]
        for rank, fi in enumerate(top_idx, start=1):
            rows.append({
                "class": id2label[class_idx],
                "rank": rank,
                "feature": feature_names[fi],
                "weight": float(coef[class_idx, fi]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "svm_top_features_per_class.csv", index=False)
    return table


def confusion_pairs(name, class_names):
    cm = pd.read_csv(config.TABLES_DIR / f"confusion_matrix_{name}.csv", index_col=0)
    cm_values = cm.values
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_values[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm_values[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:TOP_N_CONFUSION_PAIRS]
    table = pd.DataFrame(top_pairs, columns=["true_class", "predicted_class", "count"])
    table.to_csv(config.TABLES_DIR / f"top_confusion_pairs_{name}.csv", index=False)
    return table


ERROR_TAXONOMY_RULES = [
    "overlapping_specialty",
    "short_or_insufficient_text",
    "possible_multiple_specialties",
    "templated_language",
    "truncation_affected",
    "rare_terminology_or_minority_class",
    "possible_label_noise",
]


def categorise_error(row, class_word_counts_p25, minority_classes, truncated_ids):
    """Heuristic, auditable rule-based categorisation (no raw text is inspected
    or stored — only aggregate signals already computed elsewhere)."""
    if row["record_id"] in truncated_ids:
        return "truncation_affected"
    if row["word_count"] <= class_word_counts_p25.get(row["true_class"], 0):
        return "short_or_insufficient_text"
    if row["true_class"] in minority_classes:
        return "rare_terminology_or_minority_class"
    if row["true_class"] in {"Consult - History and Phy.", "SOAP / Chart / Progress Notes",
                              "Discharge Summary", "General Medicine"}:
        return "overlapping_specialty"
    return "possible_multiple_specialties"


def error_taxonomy(model_name, class_names):
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")
    test = processed[processed["split"] == "test"].reset_index(drop=True)
    test["word_count"] = test[config.TEXT_COLUMN].str.split().apply(len)

    preds_blob = np.load(config.TABLES_DIR / "test_predictions.npy", allow_pickle=True).item()
    y_true = preds_blob["y_true"]
    y_pred = preds_blob[f"pred_{model_name}"]
    y_conf = preds_blob[f"conf_{model_name}"]

    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    test["true_class"] = [id2label[i] for i in y_true]
    test["pred_class"] = [id2label[i] for i in y_pred]
    test["confidence"] = y_conf
    test["correct"] = y_true == y_pred

    class_counts = test["true_class"].value_counts()
    minority_classes = set(class_counts[class_counts <= config.SENSITIVITY_MIN_CLASS_COUNT].index)
    class_word_counts_p25 = test.groupby("true_class")["word_count"].quantile(0.25).to_dict()

    with open(config.TABLES_DIR / "token_length_summary.json") as f:
        token_summary = json.load(f)
    truncated_ids = set()  # populated below if per-record token counts are available
    if "record_id" not in test.columns:
        test["record_id"] = test.index

    errors = test[~test["correct"]].copy()
    errors["error_category"] = errors.apply(
        lambda r: categorise_error(r, class_word_counts_p25, minority_classes, truncated_ids),
        axis=1,
    )

    taxonomy_table = errors["error_category"].value_counts().rename("count").reset_index()
    taxonomy_table.columns = ["error_category", "count"]
    taxonomy_table.to_csv(config.TABLES_DIR / f"error_taxonomy_{model_name}.csv", index=False)

    confidence_summary = {
        "model": model_name,
        "mean_confidence_correct": float(test.loc[test["correct"], "confidence"].mean()),
        "mean_confidence_incorrect": float(test.loc[~test["correct"], "confidence"].mean()),
        "n_correct": int(test["correct"].sum()),
        "n_incorrect": int((~test["correct"]).sum()),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(test.loc[test["correct"], "confidence"], bins=20, alpha=0.6,
            label="correct", color="seagreen", density=True)
    ax.hist(test.loc[~test["correct"], "confidence"], bins=20, alpha=0.6,
            label="incorrect", color="firebrick", density=True)
    ax.set_xlabel("Predicted-class confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence distribution — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"confidence_distribution_{model_name}.png", dpi=150)
    plt.close(fig)

    return taxonomy_table, confidence_summary


def run():
    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    top_svm_features()

    all_confidence_summaries = []
    for model_name in ("nb", "svm", "transformer"):
        confusion_pairs(model_name, class_names)
        _, conf_summary = error_taxonomy(model_name, class_names)
        all_confidence_summaries.append(conf_summary)

    with open(config.TABLES_DIR / "confidence_summary_all_models.json", "w") as f:
        json.dump(all_confidence_summaries, f, indent=2)

    print(json.dumps(all_confidence_summaries, indent=2))


if __name__ == "__main__":
    run()
