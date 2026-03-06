# src/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.utils import set_seeds
from src.data import load_dataset, split_holdout, save_holdout_csv, LABEL_COL
from src.preprocessing import make_preprocess_pipeline
from src.models_sklearn import get_model_specs
from src.models_pytorch import TorchMLPClassifier
from src.evaluate import evaluate_proba

ART_DIR = "model_artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.joblib")
META_PATH = os.path.join(ART_DIR, "metadata.json")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cv_eval_sklearn_model(X: pd.DataFrame, y: pd.Series, estimator, needs_scaling: bool, seed: int):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    aucs, accs = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        pre = make_preprocess_pipeline(scale=needs_scaling)
        pipe = Pipeline([("preprocess", pre), ("model", estimator)])

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, proba))
        accs.append(accuracy_score(y_va, pred))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def cv_eval_torch_mlp(X: pd.DataFrame, y: pd.Series, seed: int):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    aucs, accs = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # Fit preprocessing only on training fold
        pre = make_preprocess_pipeline(scale=True)
        X_tr_p = pre.fit_transform(X_tr)
        X_va_p = pre.transform(X_va)

        clf = TorchMLPClassifier(seed=seed)
        clf.fit(X_tr_p, y_tr.values)
        proba = clf.predict_proba(X_va_p)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, proba))
        accs.append(accuracy_score(y_va, pred))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def train_final_and_save(best_name: str, X_train, y_train, X_test, y_test, seed: int, meta: dict):
    """
    Saves a single joblib artifact the Flask app can use:
      - sklearn: {"type":"sklearn_pipeline","pipeline": pipe, "metadata": meta}
      - torch:   {"type":"torch_mlp","preprocess": pre,"state_dict":..., "input_dim":..., "metadata": meta}

    Also writes metadata.json for convenience.
    """
    ensure_dir(ART_DIR)

    if best_name == "TorchMLP":
        pre = make_preprocess_pipeline(scale=True)
        Xtr_p = pre.fit_transform(X_train)
        Xte_p = pre.transform(X_test)

        clf = TorchMLPClassifier(seed=seed)
        clf.fit(Xtr_p, y_train.values)

        proba_test = clf.predict_proba(Xte_p)[:, 1]
        test_res = evaluate_proba(y_test.values, proba_test, threshold=0.5)

        meta.update({
            "best_model": best_name,
            "test_auc": test_res.auc,
            "test_accuracy": test_res.accuracy,
            "confusion_matrix": test_res.confusion_matrix,
            "threshold": 0.5,
        })

        bundle = {
            "type": "torch_mlp",
            "preprocess": pre,
            "state_dict": clf.model_.state_dict(),
            "input_dim": int(Xtr_p.shape[1]),
            "metadata": meta,
        }
        joblib.dump(bundle, MODEL_PATH)

        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    # sklearn path
    specs = {s.name: s for s in get_model_specs(seed)}
    spec = specs[best_name]
    pre = make_preprocess_pipeline(scale=spec.needs_scaling)
    pipe = Pipeline([("preprocess", pre), ("model", spec.estimator)])

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    test_res = evaluate_proba(y_test.values, proba_test, threshold=0.5)

    meta.update({
        "best_model": best_name,
        "test_auc": test_res.auc,
        "test_accuracy": test_res.accuracy,
        "confusion_matrix": test_res.confusion_matrix,
        "threshold": 0.5,
    })

    joblib.dump({"type": "sklearn_pipeline", "pipeline": pipe, "metadata": meta}, MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_csv", default="data/test_holdout_20pct.csv")
    args = parser.parse_args()

    set_seeds(args.seed)

    df = load_dataset(args.csv)

    X_train, X_test, y_train, y_test = split_holdout(df, test_size=0.2, seed=args.seed)

    # meta exists EARLY
    meta = {
        "seed": args.seed,
        "target": LABEL_COL,
        "feature_columns": list(X_train.columns),
    }

    ensure_dir(os.path.dirname(args.holdout_csv) or ".")
    save_holdout_csv(X_test, y_test, args.holdout_csv)

    results = []

    # sklearn models
    for spec in get_model_specs(args.seed):
        mean_auc, std_auc, mean_acc, std_acc = cv_eval_sklearn_model(
            X_train, y_train, spec.estimator, spec.needs_scaling, args.seed
        )
        results.append({
            "model": spec.name,
            "cv_auc_mean": mean_auc, "cv_auc_std": std_auc,
            "cv_acc_mean": mean_acc, "cv_acc_std": std_acc,
        })

    # required PyTorch MLP
    mean_auc, std_auc, mean_acc, std_acc = cv_eval_torch_mlp(X_train, y_train, args.seed)
    results.append({
        "model": "TorchMLP",
        "cv_auc_mean": mean_auc, "cv_auc_std": std_auc,
        "cv_acc_mean": mean_acc, "cv_acc_std": std_acc,
    })

    results_df = pd.DataFrame(results).sort_values(by="cv_auc_mean", ascending=False)
    ensure_dir("reports")
    results_df.to_csv("reports/model_comparison.csv", index=False)

    best_name = str(results_df.iloc[0]["model"])
    meta = train_final_and_save(best_name, X_train, y_train, X_test, y_test, args.seed, meta)

    summary = {
        "selected_by": "highest_mean_cv_auc",
        "best_model": best_name,
        "cv_table_path": "reports/model_comparison.csv",
        "test_metrics": meta,
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("Best model:", best_name)
    print("Test AUC:", meta["test_auc"], "Test Accuracy:", meta["test_accuracy"])
    print("Saved:", MODEL_PATH, META_PATH)


if __name__ == "__main__":
    main()
