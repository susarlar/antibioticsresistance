# app.py
import io
import json
import base64
import os

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

from src.inference import load_bundle, predict_df
from src.data import LABEL_COL

NON_NUMERIC_COLS = []

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_artifacts", "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model_artifacts", "metadata.json")
MODELS_DIR = os.path.join(BASE_DIR, "model_artifacts", "all_models")

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


def load_metadata():
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_cv_results():
    cv_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
    if os.path.exists(cv_path):
        return pd.read_csv(cv_path)
    return None


def _get_plt():
    """Lazy-import matplotlib to keep startup fast."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_to_base64(fig):
    plt = _get_plt()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def generate_confusion_matrix_plot(cm):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Resistant', 'Resistant'])
    ax.set_yticklabels(['Non-Resistant', 'Resistant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, weight='bold')
    plt.colorbar(im, ax=ax)
    return fig


def generate_roc_curve_plot(y_true, y_proba, model_name="Model"):
    plt = _get_plt()
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig


def generate_probability_distribution_plot(y_true, y_proba):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    proba_non_r = y_proba[y_true == 0]
    proba_r = y_proba[y_true == 1]
    ax.hist(proba_non_r, bins=50, alpha=0.6, label='Non-Resistant', color='green', density=True)
    ax.hist(proba_r, bins=50, alpha=0.6, label='Resistant', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel('Predicted Resistance Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def generate_cv_comparison_plot(cv_results):
    if cv_results is None or cv_results.empty:
        return None
    plt = _get_plt()
    df = cv_results.sort_values('cv_auc_mean', ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    y_pos = np.arange(len(df))
    ax1.barh(y_pos, df['cv_auc_mean'], xerr=df['cv_auc_std'], color='steelblue', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df['model'])
    ax1.set_xlabel('Cross-Validation AUC (mean +/- std)')
    ax1.set_title('Model Comparison - AUC')
    ax1.grid(axis='x', alpha=0.3)
    ax2.barh(y_pos, df['cv_acc_mean'], xerr=df['cv_acc_std'], color='coral', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df['model'])
    ax2.set_xlabel('Cross-Validation Accuracy (mean +/- std)')
    ax2.set_title('Model Comparison - Accuracy')
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    model_ok = os.path.exists(MODEL_PATH)
    meta = load_metadata()
    return render_template("index.html", model_ok=model_ok, meta=meta)


ABX_FEATURE_MAP = {
    'CEFEPIME': 'CEFEPIME',
    'CEFTAZIDIME': 'CEFTAZIDIME',
    'IMIPENEM': 'IMIPENEM',
    'LEVOFLOXACIN': 'LEVOFLOXACIN',
    'MEROPENEM': 'MEROPENEM',
    'PIPERACILIN+TAZOB.': 'PIPERACILIN_TAZOB',
    'SIPROFLOKSASIN': 'SIPROFLOKSASIN',
}

ABX_DISPLAY_NAMES = {
    'CEFEPIME': 'Cefepime',
    'CEFTAZIDIME': 'Ceftazidime',
    'IMIPENEM': 'Imipenem',
    'LEVOFLOXACIN': 'Levofloxacin',
    'MEROPENEM': 'Meropenem',
    'PIPERACILIN+TAZOB.': 'Pip/Tazobactam',
    'SIPROFLOKSASIN': 'Ciprofloxacin',
}

DEPT_NAMES = {
    'ICU': 'ICU',
    'Pulmonary': 'Pulmonology',
    'Oncology': 'Oncology',
    'Infectious': 'Infectious Diseases',
    'Pediatric': 'Pediatrics',
    'Other': 'Other',
}

RESULT_LABELS = {-1: 'Not tested', 0: 'S', 1: 'I', 2: 'R'}
ALL_DEPT_COLS = ['Dept_ICU', 'Dept_Infectious', 'Dept_Oncology', 'Dept_Other', 'Dept_Pediatric', 'Dept_Pulmonary']
ALL_ABX = ['CEFEPIME', 'CEFTAZIDIME', 'IMIPENEM', 'LEVOFLOXACIN',
           'MEROPENEM', 'PIPERACILIN+TAZOB.', 'SIPROFLOKSASIN']


@app.post("/predict_clinical")
def predict_clinical():
    try:
        bundle = get_bundle()
        meta = load_metadata()

        target_abx = request.form.get('target_antibiotic')
        age = float(request.form.get('age', 65))
        gender = int(request.form.get('gender', 1))
        inpatient = int(request.form.get('inpatient', 1))
        department = request.form.get('department', 'Pulmonary')

        row = {'Age': age, 'Gender': gender, 'Inpatient': inpatient, 'Year': 2025, 'Month': 1}

        for col in ALL_DEPT_COLS:
            row[col] = 0
        dept_col = f'Dept_{department}'
        if dept_col in ALL_DEPT_COLS:
            row[dept_col] = 1

        for abx in ALL_ABX:
            row[f'Target_{ABX_FEATURE_MAP[abx]}'] = 1 if abx == target_abx else 0

        known_results = {}
        for abx in ALL_ABX:
            if abx == target_abx:
                continue
            feat_name = ABX_FEATURE_MAP[abx]
            result_val = int(request.form.get(f'result_{abx}', -1))
            row[f'{feat_name}_result'] = result_val
            if result_val >= 0:
                known_results[ABX_DISPLAY_NAMES[abx]] = RESULT_LABELS[result_val]

        df = pd.DataFrame([row])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        proba, pred = predict_df(bundle, df)

        patient_summary = {
            'age': int(age),
            'gender': 'Male' if gender == 1 else 'Female',
            'setting': 'Inpatient' if inpatient == 1 else 'Outpatient',
            'department': DEPT_NAMES.get(department, department),
            'known_results': known_results,
        }

        return render_template("results.html",
            mode="clinical", proba=float(proba[0]), pred=int(pred[0]),
            meta=meta, target_antibiotic=ABX_DISPLAY_NAMES.get(target_abx, target_abx),
            patient_summary=patient_summary, table=None, metrics=None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h1>Error</h1><p>{str(e)}</p><p><a href='/'>Back</a></p>", 500


@app.post("/predict_batch")
def predict_batch():
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    try:
        bundle = get_bundle()
    except Exception as e:
        return f"<h1>Error</h1><p>Failed to load model: {str(e)}</p><p><a href='/'>Back</a></p>", 500

    meta = load_metadata()

    if "file" not in request.files:
        return render_template("results.html", mode="batch", meta=meta,
                             table=None, metrics={"error": "No file uploaded"}, proba=None, pred=None)

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return render_template("results.html", mode="batch", meta=meta,
                             table=None, metrics={"error": "Please upload a CSV"}, proba=None, pred=None)

    content = f.read()
    df = pd.read_csv(io.BytesIO(content))

    has_label = LABEL_COL in df.columns
    X = df.drop(columns=[LABEL_COL]) if has_label else df.copy()

    non_numeric = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    proba, pred = predict_df(bundle, X)

    out = X.copy()
    out["resistance_probability"] = proba
    out["prediction"] = pred

    metrics = None
    visualizations = {}

    if has_label:
        y_true = df[LABEL_COL].astype(int).values
        auc = float(roc_auc_score(y_true, proba))
        acc = float(accuracy_score(y_true, pred))
        cm = confusion_matrix(y_true, pred)
        metrics = {"auc": auc, "accuracy": acc, "confusion_matrix": cm.tolist()}

        cm_fig = generate_confusion_matrix_plot(cm)
        visualizations['confusion_matrix'] = plot_to_base64(cm_fig)
        roc_fig = generate_roc_curve_plot(y_true, proba, meta.get('best_model', 'Model'))
        visualizations['roc_curve'] = plot_to_base64(roc_fig)
        prob_fig = generate_probability_distribution_plot(y_true, proba)
        visualizations['prob_distribution'] = plot_to_base64(prob_fig)

    preview = out.head(50).to_html(classes="table table-striped", index=False)

    return render_template("results.html", mode="batch", proba=None, pred=None,
                         meta=meta, table=preview, metrics=metrics, visualizations=visualizations)


@app.get("/model_comparison")
def model_comparison():
    cv_results = get_cv_results()
    meta = load_metadata()

    visualizations = {}
    if cv_results is not None:
        cv_fig = generate_cv_comparison_plot(cv_results)
        if cv_fig:
            visualizations['cv_comparison'] = plot_to_base64(cv_fig)
        cv_table = cv_results.to_html(classes="table table-striped", index=False)
    else:
        cv_table = "<p>No cross-validation results available.</p>"

    return render_template("model_comparison.html",
                         meta=meta, cv_table=cv_table, visualizations=visualizations)


if __name__ == "__main__":
    app.run(debug=True)
