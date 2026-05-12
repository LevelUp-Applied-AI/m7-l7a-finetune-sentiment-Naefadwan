"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    # TODO: iterate texts in batches
    # TODO: tokenize each batch with truncation, max_length=128, padding=True, return_tensors='pt'
    # TODO: forward pass under torch.no_grad()
    # TODO: softmax over the last dim
    # TODO: argmax to get class indices
    # TODO: collect into numpy arrays of shape (N,) and (N, num_classes); return both 
    preds = []
    probs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            preds.append(preds) 
            probs.append(probs)
    return np.array(preds), np.array(probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    # TODO: compute true positives / false positives / false negatives per class
    # TODO: precision = TP / (TP + FP); guard divide-by-zero
    # TODO: recall = TP / (TP + FN)
    # TODO: f1 = 2 * P * R / (P + R)
    # TODO: accuracy = sum(y_pred == y_true) / N
    # TODO: macro-F1 = mean of per-class f1 scores
    # TODO: assemble and return the dict
    true_positives = {i: 0 for i in range(len(y_true))}
    false_positives = {i: 0 for i in range(len(y_true))}
    false_negatives = {i: 0 for i in range(len(y_true))}
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            true_positives[y_true[i]] += 1
        else:
            false_positives[y_true[i]] += 1
            false_negatives[y_pred[i]] += 1
    precision = {i: true_positives[i] / (true_positives[i] + false_positives[i]) for i in range(len(y_true))}
    recall = {i: true_positives[i] / (true_positives[i] + false_negatives[i]) for i in range(len(y_true))}
    f1 = {i: 2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(y_true))}
    accuracy = sum(y_pred == y_true) / len(y_true)
    macro_f1 = sum(f1.values()) / len(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {i: {"precision": precision[i], "recall": recall[i], "f1": f1[i]} for i in range(len(y_true))},
    }
