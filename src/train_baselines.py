"""Traditional NLP baselines: TF-IDF + Multinomial NB, TF-IDF (word+char) + Linear SVM.

Implements plan section 6: both baselines are scikit-learn Pipelines so TF-IDF
fitting happens inside each CV training fold only (no leakage), hyperparameters
are chosen by 5-fold stratified CV macro-F1 on the training split, and the
selected Linear SVM is saved directly so evaluation uses the same decision rule
that won cross-validation.
"""

import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config


def load_split(split_name: str) -> pd.DataFrame:
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")
    return processed[processed["split"] == split_name].reset_index(drop=True)


def build_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode")),
        ("clf", MultinomialNB()),
    ])


def build_svm_pipeline() -> Pipeline:
    """Word (1-2 gram) + character (3-5 gram, word-boundary) TF-IDF, per the plan's
    'word/character TF-IDF' search-space entry for the Linear SVM baseline."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                  ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([
        ("tfidf", features),
        ("clf", LinearSVC(max_iter=10000)),
    ])


NB_GRID = {
    "tfidf__ngram_range": config.NB_PARAM_GRID["tfidf__ngram_range"],
    "tfidf__min_df": config.NB_PARAM_GRID["tfidf__min_df"],
    "tfidf__max_df": config.NB_PARAM_GRID["tfidf__max_df"],
    "clf__alpha": config.NB_PARAM_GRID["clf__alpha"],
}

SVM_GRID = {
    "tfidf__word__min_df": config.SVM_PARAM_GRID["tfidf__min_df"],
    "tfidf__word__ngram_range": config.SVM_PARAM_GRID["tfidf__ngram_range"],
    "clf__C": config.SVM_PARAM_GRID["clf__C"],
    "clf__class_weight": config.SVM_PARAM_GRID["clf__class_weight"],
}


def run_search(pipeline, grid, X, y, name):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    search = GridSearchCV(
        pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True, verbose=1
    )
    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    cv_table = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    cv_table.to_csv(config.TABLES_DIR / f"{name}_cv_results.csv", index=False)

    summary = {
        "model": name,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in search.best_params_.items()},
        "best_cv_macro_f1": float(search.best_score_),
        "cv_folds": config.CV_FOLDS,
        "search_time_seconds": round(elapsed, 1),
    }
    with open(config.TABLES_DIR / f"{name}_best_params.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return search


def run():
    train = load_split("train")
    X_train, y_train = train[config.TEXT_COLUMN], train["label_id"]

    nb_search = run_search(build_nb_pipeline(), NB_GRID, X_train, y_train, "nb")
    joblib.dump(nb_search.best_estimator_, config.BASELINE_MODELS_DIR / "nb_pipeline.joblib")

    svm_search = run_search(build_svm_pipeline(), SVM_GRID, X_train, y_train, "svm")
    joblib.dump(svm_search.best_estimator_, config.BASELINE_MODELS_DIR / "svm_pipeline.joblib")

    print("Saved nb_pipeline.joblib and svm_pipeline.joblib.")


if __name__ == "__main__":
    run()
