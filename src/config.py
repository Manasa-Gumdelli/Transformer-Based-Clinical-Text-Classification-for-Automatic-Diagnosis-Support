"""Central configuration: paths, seeds, label scope, model settings."""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "mtsamples.csv"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

for d in (PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, CONFIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ----------------------------------------------------
RANDOM_SEED = 42

# --- Task scope -----------------------------------------------------------
# Primary experiment: retain specialties with >= MIN_CLASS_COUNT records.
MIN_CLASS_COUNT = 100
# Exact duplicate transcriptions in mtsamples frequently carry conflicting
# specialty labels. Collapsing them makes the supervised target identifiable
# from transcription-only input while preserving the leakage constraint.
COLLAPSE_DUPLICATE_TRANSCRIPTIONS = True

# Secondary sensitivity experiment (S2): lower threshold, more classes.
SENSITIVITY_MIN_CLASS_COUNT = 50

TEXT_COLUMN = "transcription"
LABEL_COLUMN = "medical_specialty"
# Columns intentionally excluded from the primary model to prevent leakage.
LEAKY_COLUMNS = ["keywords", "sample_name", "description"]
DROP_COLUMNS = ["Unnamed: 0"]

# --- Split configuration -------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Baseline model search spaces (see plan section 6) --------------------
NB_PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [0.90, 0.95],
    "clf__alpha": [0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
}

SVM_PARAM_GRID = {
    "tfidf__ngram_range": [(1, 2), (1, 3)],
    "tfidf__min_df": [1, 2],
    "clf__C": [1, 2, 4],
    "clf__class_weight": [None, "balanced"],
}

CV_FOLDS = 5

# --- Transformer configuration (see plan section 7) -----------------------
TRANSFORMER_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TRANSFORMER_FALLBACK_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# Token-length audit (outputs/tables/token_length_summary.json) showed 89.7% of
# retained transcriptions exceed 256 tokens and 64.2% exceed the 512 hard limit,
# so the primary run uses the maximum single-sequence length (512) rather than
# the plan's default 256, per plan 7.3 ("compare ... only if justified by the
# token-length analysis"). Residual truncation beyond 512 is reported, not hidden.
MAX_SEQ_LENGTH = 512
TRANSFORMER_CHUNK_STRIDE = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
# The cleaned, de-duplicated dataset is smaller, and the transformer is a
# comparison model rather than the expected best scorer. Three epochs keeps the
# full pipeline practical on a laptop/MPS run while validation macro-F1 still
# selects the best checkpoint.
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

TRANSFORMER_CHECKPOINT_DIR = MODELS_DIR / "transformer"
BASELINE_MODELS_DIR = MODELS_DIR / "baselines"
BASELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Dataset provenance (for manifest / ethics reporting) -----------------
DATASET_SOURCE = "Kaggle: Medical Transcriptions (mtsamples.csv)"
DATASET_LICENCE_NOTE = (
    "Publicly released, de-identified sample medical transcriptions distributed via "
    "Kaggle for research/education. No re-identification is attempted. Raw text is "
    "excluded from version control and must not be reproduced verbatim in reports."
)
