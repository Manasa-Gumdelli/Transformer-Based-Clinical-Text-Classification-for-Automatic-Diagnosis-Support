"""Fine-tune a domain-adapted transformer (Bio_ClinicalBERT) for medical-specialty
classification (plan section 7). Uses the same fixed 70/15/15 split as the
baselines; validation macro-F1 drives checkpoint selection and early stopping.
The held-out test set is left untouched until src/evaluate.py runs once.
"""

import json
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import config
from .transformer_chunks import aggregate_logits_by_doc, build_chunked_dataset


def get_device_info() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_datasets(tokenizer):
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")

    def to_hf(split_name):
        subset = processed[processed["split"] == split_name].reset_index(drop=True)
        return build_chunked_dataset(subset, tokenizer)

    return to_hf("train"), to_hf("val")


def compute_metrics(eval_pred, doc_ids=None):
    logits, labels = eval_pred
    if doc_ids is not None:
        logits, labels = aggregate_logits_by_doc(logits, labels, doc_ids)
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


class DocumentChunkTrainer(Trainer):
    """Trainer variant that keeps doc_id for document-level validation metrics."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("doc_id", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = dict(inputs)
        inputs.pop("doc_id", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def run():
    set_seed(config.RANDOM_SEED)
    device = get_device_info()

    with open(config.PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    id2label = {v: k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map,
        local_files_only=True,
    )
    n_params = sum(p.numel() for p in model.parameters())

    train_ds, val_ds = load_datasets(tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    steps_per_epoch = max(1, len(train_ds) // config.TRAIN_BATCH_SIZE)
    args = TrainingArguments(
        output_dir=str(config.TRANSFORMER_CHECKPOINT_DIR / "runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 4),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=config.RANDOM_SEED,
        report_to=[],
        # fp16/bf16 autocast is not reliably supported by the Trainer on the MPS
        # backend in this transformers version; mixed precision is left off here
        # and only enabled automatically when running on CUDA hardware.
        fp16=(device == "cuda"),
        use_mps_device=(device == "mps"),
        # MPS can produce non-contiguous parameter tensors that safetensors'
        # save path rejects; torch.save (save_safetensors=False) has no such
        # restriction and is equally reproducible for this project's purposes.
        save_safetensors=False,
    )

    val_doc_ids = np.asarray(val_ds["doc_id"])

    trainer = DocumentChunkTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, val_doc_ids),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )

    start = time.time()
    trainer.train()
    training_time = time.time() - start

    final_model_dir = config.TRANSFORMER_CHECKPOINT_DIR / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(config.TABLES_DIR / "transformer_training_log.csv", index=False)

    best_val_metrics = trainer.evaluate(val_ds)
    processed = pd.read_csv(config.PROCESSED_DIR / "dataset.csv")

    resource_summary = {
        "model_identifier": config.TRANSFORMER_MODEL_NAME,
        "retrieved_date": str(date.today()),
        "n_parameters": int(n_params),
        "tokenizer": tokenizer.__class__.__name__,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "chunk_stride": config.TRANSFORMER_CHUNK_STRIDE,
        "device": device,
        "train_documents": int((processed["split"] == "train").sum()),
        "val_documents": int((processed["split"] == "val").sum()),
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "epochs_configured": config.NUM_EPOCHS,
        "epochs_completed": trainer.state.epoch,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "training_time_seconds": round(training_time, 1),
        "best_val_f1_macro": best_val_metrics["eval_f1_macro"],
        "final_checkpoint_path": str(final_model_dir.relative_to(config.ROOT_DIR)),
    }
    with open(config.TABLES_DIR / "transformer_resource_usage.json", "w") as f:
        json.dump(resource_summary, f, indent=2)

    print(json.dumps(resource_summary, indent=2))
    return resource_summary


if __name__ == "__main__":
    run()
