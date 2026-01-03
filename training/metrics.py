"""
Evaluation Metrics Module
=========================

Comprehensive metrics for binary classification evaluation.

METRIC DOCUMENTATION:

1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
   - Overall proportion of correct predictions

2. Precision = TP / (TP + FP)
   - Of all positive predictions, how many are correct?

3. Recall (Sensitivity) = TP / (TP + FN)
   - Of all actual positives, how many did we catch?
   - CRITICAL for safety: High recall = fewer missed crashes

4. Specificity = TN / (TN + FP)
   - Of all actual negatives, how many did we correctly identify?

5. F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall

6. AUROC (Area Under ROC Curve)
   - Probability that a random positive ranks higher than a random negative
   - Threshold-independent metric

7. Average Precision (AP)
   - Area under the precision-recall curve
   - Better for imbalanced datasets

8. MCC (Matthews Correlation Coefficient)
   - Balanced measure that accounts for all four confusion matrix values
   - Range: -1 to +1 (0 = random)
   - Formula: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
"""

from typing import Dict, Optional, Tuple

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (N,) with values in {0, 1}
        y_pred_probs: Predicted probabilities (N,) in [0, 1]
        threshold: Decision threshold (default: 0.5)
        
    Returns:
        Dictionary with all computed metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred_probs = np.asarray(y_pred_probs).flatten()
    
    # Ensure probabilities are in valid range
    y_pred_probs = np.clip(y_pred_probs, 0, 1)
    
    # Get binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Compute confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    total = len(y_true)
    
    # Basic metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # Matthews Correlation Coefficient
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    
    # AUROC
    auroc = compute_auroc(y_true, y_pred_probs)
    
    # Average Precision
    ap = compute_average_precision(y_true, y_pred_probs)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "auroc": float(auroc),
        "average_precision": float(ap),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve using trapezoidal rule.
    
    Args:
        y_true: Binary ground truth
        y_scores: Predicted scores/probabilities
        
    Returns:
        AUROC value in [0, 1]
    """
    # Handle edge cases
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return chance level
    
    # Sort by scores descending
    desc_order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_order]
    y_scores_sorted = y_scores[desc_order]
    
    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0, 0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Compute area using trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    
    return auroc


def compute_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Average Precision (area under precision-recall curve).
    
    Args:
        y_true: Binary ground truth
        y_scores: Predicted scores/probabilities
        
    Returns:
        Average Precision value
    """
    n_pos = np.sum(y_true == 1)
    
    if n_pos == 0:
        return 0.0
    
    # Sort by scores descending
    desc_order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_order]
    
    # Compute precision at each threshold
    tps = np.cumsum(y_true_sorted)
    precision = tps / np.arange(1, len(y_true_sorted) + 1)
    
    # Compute recall changes
    recall_delta = y_true_sorted / n_pos
    
    # Average precision
    ap = np.sum(precision * recall_delta)
    
    return ap


def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix components.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        
    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    return int(tp), int(tn), int(fp), int(fn)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    criterion: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal decision threshold based on criterion.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Predicted probabilities
        criterion: 'f1', 'accuracy', 'youden' (Youden's J statistic)
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_value = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        metrics = compute_metrics(y_true, y_pred_probs, threshold)
        
        if criterion == "f1":
            value = metrics["f1"]
        elif criterion == "accuracy":
            value = metrics["accuracy"]
        elif criterion == "youden":
            value = metrics["recall"] + metrics["specificity"] - 1
        else:
            value = metrics["f1"]
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    return best_threshold, best_value


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary as string."""
    parts = []
    
    for key in ["accuracy", "precision", "recall", "f1", "auroc"]:
        if key in metrics:
            name = f"{prefix}{key}" if prefix else key
            parts.append(f"{name}={metrics[key]:.4f}")
    
    return " | ".join(parts)
