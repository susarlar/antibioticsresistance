# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COL = "Label"

DROP_NON_NUMERIC = True

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing target column '{LABEL_COL}'")

    # Drop known non-numeric metadata columns (and any other non-numeric features)
    feature_df = df.drop(columns=[LABEL_COL])
    non_numeric = feature_df.select_dtypes(exclude="number").columns.tolist()

    if non_numeric and DROP_NON_NUMERIC:
        print("Dropping non-numeric columns:", non_numeric)
        df = df.drop(columns=non_numeric)

    return df

def split_holdout(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def save_holdout_csv(X_test: pd.DataFrame, y_test: pd.Series, out_path: str) -> None:
    out = X_test.copy()
    out[LABEL_COL] = y_test.astype(int).values
    out.to_csv(out_path, index=False)


def split_holdout(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test

def save_holdout_csv(X_test: pd.DataFrame, y_test: pd.Series, out_path: str) -> None:
    out = X_test.copy()
    out[LABEL_COL] = y_test.astype(int).values
    out.to_csv(out_path, index=False)
