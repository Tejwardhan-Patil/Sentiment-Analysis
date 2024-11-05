import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def sentiment_accuracy(y_true, y_pred):
    """
    Computes the accuracy for sentiment classification.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.

    Returns:
        Accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def sentiment_f1_score(y_true, y_pred, average='weighted'):
    """
    Computes the F1 score for sentiment classification.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.
        average: Strategy for multi-class F1 score computation ('micro', 'macro', 'weighted').

    Returns:
        F1 score.
    """
    return f1_score(y_true, y_pred, average=average)

def sentiment_precision(y_true, y_pred, average='weighted'):
    """
    Computes the precision for sentiment classification.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.
        average: Strategy for multi-class precision computation ('micro', 'macro', 'weighted').

    Returns:
        Precision score.
    """
    return precision_score(y_true, y_pred, average=average)

def sentiment_recall(y_true, y_pred, average='weighted'):
    """
    Computes the recall for sentiment classification.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.
        average: Strategy for multi-class recall computation ('micro', 'macro', 'weighted').

    Returns:
        Recall score.
    """
    return recall_score(y_true, y_pred, average=average)

def sentiment_confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes the confusion matrix for sentiment classification.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.
        labels: List of sentiment labels for ordering in the confusion matrix.

    Returns:
        Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)

def compute_all_metrics(y_true, y_pred, labels=None, average='weighted'):
    """
    Computes all major metrics for sentiment classification, including accuracy, F1, precision, recall, and confusion matrix.

    Args:
        y_true: Ground truth sentiment labels.
        y_pred: Predicted sentiment labels.
        labels: List of sentiment labels for confusion matrix.
        average: Strategy for multi-class metric computation ('micro', 'macro', 'weighted').

    Returns:
        Dictionary of all computed metrics.
    """
    metrics = {
        'accuracy': sentiment_accuracy(y_true, y_pred),
        'f1_score': sentiment_f1_score(y_true, y_pred, average=average),
        'precision': sentiment_precision(y_true, y_pred, average=average),
        'recall': sentiment_recall(y_true, y_pred, average=average),
        'confusion_matrix': sentiment_confusion_matrix(y_true, y_pred, labels=labels)
    }
    return metrics