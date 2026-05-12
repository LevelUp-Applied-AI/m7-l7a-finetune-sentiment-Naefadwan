"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual PyTorch inference. No Trainer.predict.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def compute_classification_report_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 using
    only numpy. No sklearn metric helpers, no Hugging Face evaluate.

    Returns: {
      "accuracy": float,
      "macro_f1": float,
      "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
    }
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    f1_scores = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        f1_scores.append(f1)

    accuracy = np.mean(y_true == y_pred)
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }
