# src/models_sklearn.py
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool

def get_model_specs(seed: int = 42):
    # Required: Logistic Regression, Decision Tree, Random Forest
    # Additional: GradientBoosting, HistGradientBoosting, XGBoost, LightGBM
    return [
        ModelSpec(
            "LogReg",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            True,
        ),
        ModelSpec("DecisionTree", DecisionTreeClassifier(random_state=seed), False),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
            False,
        ),
        ModelSpec("GradBoost", GradientBoostingClassifier(random_state=seed), False),
        ModelSpec("HistGradBoost", HistGradientBoostingClassifier(random_state=seed), False),
        ModelSpec(
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            ),
            False,
        ),
        ModelSpec(
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1
            ),
            False,
        ),
    ]
