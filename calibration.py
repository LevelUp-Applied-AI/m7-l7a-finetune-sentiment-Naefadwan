"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; for each bin, compute
    the empirical accuracy.

    Binning convention: bin edges are np.linspace(0, 1, n_bins + 1).
    A probability p falls in bin i if edges[i] <= p < edges[i+1], with
    the last bin inclusive on the right.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all of length n_bins.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Get max probability and predicted class for each sample
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Handle the last bin inclusive on the right
        if i == n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i+1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i+1])
            
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            accuracies[i] = np.mean(preds[mask] == y_true[mask])
            
    return centers, accuracies, counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    N = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i+1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i+1])
            
        count = np.sum(mask)
        if count > 0:
            accuracy = np.mean(preds[mask] == y_true[mask])
            confidence = np.mean(confidences[mask])
            ece += (count / N) * np.abs(accuracy - confidence)
            
    return float(ece)



def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
