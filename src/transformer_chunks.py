"""Utilities for document-level transformer training over long notes.

Clinical notes frequently exceed BERT's 512-token limit. These helpers split a
document into overlapping token windows, then aggregate chunk logits back to one
document prediction for validation and held-out testing.
"""

from collections import OrderedDict

import numpy as np
from datasets import Dataset

from . import config


def iter_text_chunks(tokenizer, text: str, max_length=None, stride=None):
    max_length = max_length or config.MAX_SEQ_LENGTH
    stride = config.TRANSFORMER_CHUNK_STRIDE if stride is None else stride
    content_window = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    step = max(1, content_window - stride)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        token_ids = [tokenizer.unk_token_id]

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + content_window]
        encoded = tokenizer.prepare_for_model(
            chunk_ids,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        yield encoded
        if start + content_window >= len(token_ids):
            break


def build_chunked_dataset(frame, tokenizer):
    rows = []
    for doc_id, row in enumerate(frame.itertuples(index=False)):
        text = getattr(row, config.TEXT_COLUMN)
        label = int(getattr(row, "label_id"))
        for chunk in iter_text_chunks(tokenizer, text):
            chunk["label"] = label
            chunk["doc_id"] = doc_id
            rows.append(chunk)
    return Dataset.from_list(rows)


def aggregate_logits_by_doc(logits, labels, doc_ids):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    doc_ids = np.asarray(doc_ids)
    grouped = OrderedDict()

    for idx, doc_id in enumerate(doc_ids):
        grouped.setdefault(int(doc_id), {"logits": [], "label": int(labels[idx])})
        grouped[int(doc_id)]["logits"].append(logits[idx])

    doc_logits = []
    doc_labels = []
    for item in grouped.values():
        doc_logits.append(np.mean(item["logits"], axis=0))
        doc_labels.append(item["label"])

    return np.asarray(doc_logits), np.asarray(doc_labels)
