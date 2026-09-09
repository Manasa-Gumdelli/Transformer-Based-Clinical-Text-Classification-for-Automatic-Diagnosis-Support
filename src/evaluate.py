"""Single held-out test evaluation for B1 (NB), B2 (SVM) and T1 (transformer),
per plan sections 8.2/8.3. Run once, after all model/hyperparameter choices are
frozen on train/validation data.
"""

import json
import time
import tracemalloc

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .transformer_chunks import aggregate_logits_by_doc, iter_text_chunks

N_BOOTSTRAP = 2000


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def load_test_set():
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")
    test = processed[processed["split"] == "test"].reset_index(drop=True)
    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}
    class_names = [id2label[i] for i in range(len(id2label))]
    return test, class_names


def predict_sklearn(pipeline, texts):
    tracemalloc.start()
    start = time.time()
    preds = pipeline.predict(texts)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        proba = softmax(scores)
    else:
        proba = None
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    confidences = proba.max(axis=1) if proba is not None else None
    return preds, confidences, elapsed, peak / (1024 ** 2)


def predict_transformer(model_dir, texts, batch_size=16):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    chunk_features = []
    chunk_doc_ids = []
    for doc_id, text in enumerate(texts):
        for chunk in iter_text_chunks(tokenizer, text):
            chunk_features.append(chunk)
            chunk_doc_ids.append(doc_id)

    all_logits = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(chunk_features), batch_size):
            batch = chunk_features[i:i + batch_size]
            enc = tokenizer.pad(batch, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            all_logits.append(logits.cpu().numpy())
    elapsed = time.time() - start
    chunk_logits = np.vstack(all_logits)
    dummy_labels = np.zeros(len(chunk_logits), dtype=int)
    doc_logits, _ = aggregate_logits_by_doc(chunk_logits, dummy_labels, chunk_doc_ids)
    probs = softmax(doc_logits)
    all_preds = probs.argmax(axis=1)
    all_conf = probs.max(axis=1)
    n_params = sum(p.numel() for p in model.parameters())
    peak_mb = (
        torch.mps.current_allocated_memory() / (1024 ** 2)
        if device.type == "mps" else 0.0
    )
    return np.array(all_preds), np.array(all_conf), elapsed, peak_mb, n_params


def compute_metrics(y_true, y_pred, class_names):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def bootstrap_ci_macro_f1(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return float(lower), float(upper), scores


def paired_bootstrap_comparison(y_true, pred_a, pred_b, name_a, name_b,
                                 n_boot=N_BOOTSTRAP, seed=config.RANDOM_SEED):
    """Proportion of bootstrap resamples where model A's macro-F1 exceeds model B's."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        diffs[b] = f1_a - f1_b
    p_a_better = float((diffs > 0).mean())
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": f"{name_a}_minus_{name_b}",
        "mean_diff": float(diffs.mean()),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "p_a_better_than_b": p_a_better,
    }


def mcnemar_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true)
    correct_b = (np.asarray(pred_b) == y_true)
    b = int(((correct_a) & (~correct_b)).sum())  # a right, b wrong
    c = int(((~correct_a) & (correct_b)).sum())  # a wrong, b right
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


def save_confusion_matrix(y_true, y_pred, class_names, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {name}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        config.TABLES_DIR / f"confusion_matrix_{name}.csv"
    )
    return cm


def per_class_table(y_true, y_pred, class_names, name):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    table = pd.DataFrame({
        "class": class_names, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    }).sort_values("support", ascending=False)
    table.to_csv(config.TABLES_DIR / f"per_class_metrics_{name}.csv", index=False)
    return table


def save_model_comparison_figures(comparison_table: pd.DataFrame):
    labels = {
        "nb": "Naive Bayes",
        "svm": "Linear SVM",
        "transformer": "Bio-ClinicalBERT",
    }
    colors = {
        "nb": "#4C78A8",
        "svm": "#F58518",
        "transformer": "#54A24B",
    }
    model_names = comparison_table.index.tolist()
    display_names = [labels.get(name, name) for name in model_names]
    model_colors = [colors.get(name, "#777777") for name in model_names]

    metric_cols = ["accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro-F1", "Weighted-F1"]
    metric_values = comparison_table[metric_cols].astype(float)
    x = np.arange(len(metric_cols))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, model_name in enumerate(model_names):
        values = metric_values.loc[model_name].to_numpy()
        pos = x + (offset - (len(model_names) - 1) / 2) * width
        bars = ax.bar(
            pos,
            values,
            width,
            label=labels.get(model_name, model_name),
            color=colors.get(model_name, "#777777"),
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test performance by model")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison_metrics.png", dpi=150)
    plt.close(fig)

    macro_scores = comparison_table["macro_f1"].astype(float).to_numpy()
    ci_lower = []
    ci_upper = []
    for model_name in model_names:
        ci = comparison_table.loc[model_name, "macro_f1_ci95"]
        if isinstance(ci, str):
            ci = json.loads(ci)
        lower, upper = ci
        score = float(comparison_table.loc[model_name, "macro_f1"])
        ci_lower.append(score - lower)
        ci_upper.append(upper - score)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(display_names, macro_scores, color=model_colors)
    ax.errorbar(
        display_names,
        macro_scores,
        yerr=np.array([ci_lower, ci_upper]),
        fmt="none",
        ecolor="#333333",
        capsize=5,
        linewidth=1.2,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 comparison with 95% bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison_macro_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    inference_times = comparison_table["inference_time_seconds"].astype(float).to_numpy()
    bars = ax.bar(display_names, inference_times, color=model_colors)
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=9)
    ax.set_ylabel("Seconds")
    ax.set_title("Held-out test inference time")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison_inference_time.png", dpi=150)
    plt.close(fig)


def save_per_class_f1_comparison(class_names):
    rows = []
    labels = {
        "nb": "Naive Bayes",
        "svm": "Linear SVM",
        "transformer": "Bio-ClinicalBERT",
    }
    colors = {
        "Naive Bayes": "#4C78A8",
        "Linear SVM": "#F58518",
        "Bio-ClinicalBERT": "#54A24B",
    }
    hue_order = ["Naive Bayes", "Linear SVM", "Bio-ClinicalBERT"]
    for model_name in ("nb", "svm", "transformer"):
        table = pd.read_csv(config.TABLES_DIR / f"per_class_metrics_{model_name}.csv")
        table["model"] = labels[model_name]
        rows.append(table[["class", "f1", "model"]])
    per_class = pd.concat(rows, ignore_index=True)
    per_class["class"] = pd.Categorical(per_class["class"], categories=class_names, ordered=True)
    per_class = per_class.sort_values("class")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=per_class,
        x="class",
        y="f1",
        hue="model",
        hue_order=hue_order,
        palette=colors,
        ax=ax,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 comparison")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "per_class_f1_comparison.png", dpi=150)
    plt.close(fig)


def run():
    test, class_names = load_test_set()
    y_true = test["label_id"].to_numpy()
    texts = test[config.TEXT_COLUMN].tolist()

    results = {}
    predictions = {}
    confidences = {}

    nb_pipeline = joblib.load(config.BASELINE_MODELS_DIR / "nb_pipeline.joblib")
    nb_preds, nb_conf, nb_time, nb_mem = predict_sklearn(nb_pipeline, texts)
    predictions["nb"] = nb_preds
    confidences["nb"] = nb_conf
    results["nb"] = {
        **compute_metrics(y_true, nb_preds, class_names),
        "inference_time_seconds": nb_time,
        "peak_memory_mb": nb_mem,
    }
    save_confusion_matrix(y_true, nb_preds, class_names, "nb")
    per_class_table(y_true, nb_preds, class_names, "nb")

    svm_pipeline = joblib.load(config.BASELINE_MODELS_DIR / "svm_pipeline.joblib")
    svm_preds, svm_conf, svm_time, svm_mem = predict_sklearn(svm_pipeline, texts)
    predictions["svm"] = svm_preds
    confidences["svm"] = svm_conf
    results["svm"] = {
        **compute_metrics(y_true, svm_preds, class_names),
        "inference_time_seconds": svm_time,
        "peak_memory_mb": svm_mem,
    }
    save_confusion_matrix(y_true, svm_preds, class_names, "svm")
    per_class_table(y_true, svm_preds, class_names, "svm")

    transformer_dir = config.TRANSFORMER_CHECKPOINT_DIR / "final_model"
    t_preds, t_conf, t_time, t_mem, t_params = predict_transformer(str(transformer_dir), texts)
    predictions["transformer"] = t_preds
    confidences["transformer"] = t_conf
    results["transformer"] = {
        **compute_metrics(y_true, t_preds, class_names),
        "inference_time_seconds": t_time,
        "peak_memory_mb": t_mem,
        "n_parameters": int(t_params),
    }
    save_confusion_matrix(y_true, t_preds, class_names, "transformer")
    per_class_table(y_true, t_preds, class_names, "transformer")

    for name in results:
        lower, upper, _ = bootstrap_ci_macro_f1(y_true, predictions[name])
        results[name]["macro_f1_ci95"] = [lower, upper]

    comparison_table = pd.DataFrame(results).T
    comparison_table.to_csv(config.TABLES_DIR / "model_comparison.csv")
    save_model_comparison_figures(comparison_table)
    save_per_class_f1_comparison(class_names)

    pairwise = []
    model_names = list(results.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            pairwise.append(paired_bootstrap_comparison(
                y_true, predictions[a], predictions[b], a, b
            ))
            mcnemar = mcnemar_test(y_true, predictions[a], predictions[b])
            pairwise[-1]["mcnemar"] = mcnemar
    with open(config.TABLES_DIR / "pairwise_model_comparison.json", "w") as f:
        json.dump(pairwise, f, indent=2)

    np.save(config.TABLES_DIR / "test_predictions.npy", {
        "y_true": y_true,
        **{f"pred_{k}": v for k, v in predictions.items()},
        **{f"conf_{k}": v for k, v in confidences.items()},
    }, allow_pickle=True)

    print(comparison_table)
    print(json.dumps(pairwise, indent=2))
    return comparison_table, pairwise


if __name__ == "__main__":
    run()
