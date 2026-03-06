# src/evaluate.py
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

@dataclass
class EvalResult:
    auc: float
    accuracy: float
    confusion_matrix: list  # 2x2 list

def evaluate_proba(y_true, y_proba, threshold: float = 0.5) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    auc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return EvalResult(auc=auc, accuracy=acc, confusion_matrix=cm)
