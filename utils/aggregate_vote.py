import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score

def aggregate_predictions(predictions, labels, group_size=5, threshold=0.5):
    """
    Aggregate predictions by averaging all possible combinations of `group_size`
    within each label class (positive and negative), then apply threshold.
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Separate positive and negative samples
    pos_scores = predictions[labels == 1]
    neg_scores = predictions[labels == 0]

    def generate_combinations(scores, label, group_size, threshold):
        combs = list(combinations(range(len(scores)), group_size))
        avg_scores = np.array([np.mean(scores[list(c)]) for c in combs])
        preds = (avg_scores > threshold).astype(int)
        gt_labels = np.full(len(combs), label)
        return avg_scores, preds, gt_labels

    pos_avg_scores, pos_preds, pos_labels = generate_combinations(pos_scores, 1, group_size, threshold)
    neg_avg_scores, neg_preds, neg_labels = generate_combinations(neg_scores, 0, group_size, threshold)
    # print("Positive Combinations:", len(pos_avg_scores), "Negative Combinations:", len(neg_avg_scores))
    # print("Positive Scores:", pos_avg_scores, "Negative Scores:", neg_avg_scores)
    # print("Positive Predictions:", pos_preds, "Negative Predictions:", neg_preds)
    # print("Positive Labels:", pos_labels, "Negative Labels:", neg_labels)

    # Combine results
    all_scores = np.concatenate([pos_avg_scores, neg_avg_scores])
    all_preds = np.concatenate([pos_preds, neg_preds])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Metrics
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    ap_score = average_precision_score(all_labels, all_scores)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    acc = 100 * (tp + tn) / (tp + tn + fp + fn)

    return {
        "group_size": group_size,
        "threshold": threshold,
        "accuracy": acc,
        "auc": auc_score,
        "avg_precision": ap_score,
        "far": far,
        "frr": frr,
        "conf_matrix": [[tn, fp], [fn, tp]],
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall
    }
