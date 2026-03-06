# src/inference.py
import numpy as np
import pandas as pd
import joblib

LABEL_COL = "Label"

# Known non-numeric columns to drop (legacy, kept for compatibility)
NON_NUMERIC_COLS = []


def load_bundle(model_path: str):
    obj = joblib.load(model_path)
    # If old artifact saved only pipeline, wrap it
    if isinstance(obj, dict) and ("pipeline" in obj or obj.get("type") == "torch_mlp"):
        return obj
    return {"type": "sklearn_pipeline", "pipeline": obj, "metadata": {}}


def _align_to_feature_cols(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])
    
    # Drop non-numeric columns if present
    for col in NON_NUMERIC_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Also drop any remaining non-numeric columns
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)

    # Add missing columns as NaN (imputer will handle)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Drop extras + enforce order
    df = df[feature_cols]
    return df


def predict_df(bundle, df: pd.DataFrame):
    meta = bundle.get("metadata", {}) or {}
    feature_cols = meta.get("feature_columns")

    # Align columns if we have a schema
    if feature_cols:
        df = _align_to_feature_cols(df, feature_cols)
    else:
        if LABEL_COL in df.columns:
            df = df.drop(columns=[LABEL_COL])

    if df.shape[1] == 0:
        raise ValueError("No feature columns found. Check uploaded CSV columns or form payload.")

    # sklearn pipeline
    if bundle.get("type") != "torch_mlp":
        pipe = bundle["pipeline"]
        proba = pipe.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return proba, pred

    # Torch path
    import torch
    import numpy as np
    from src.models_pytorch import MLP
    
    pre = bundle["preprocess"]
    state_dict = bundle["state_dict"]
    input_dim = bundle["input_dim"]
    
    # Transform data
    df_transformed = pre.transform(df)
    
    # Load model
    model = MLP(input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.from_numpy(df_transformed.astype(np.float32))
        logits = model(X_tensor).numpy().reshape(-1)
        proba = 1 / (1 + np.exp(-logits))
    
    pred = (proba >= 0.5).astype(int)
    return proba, pred
