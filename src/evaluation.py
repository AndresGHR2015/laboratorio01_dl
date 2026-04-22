import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from balancing import apply_smote
from config import CV_SPLITS, RANDOM_STATE
from feature_selection import select_k_best_features

def compute_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1 (macro)": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def evaluate_models_for_target(
    X: np.ndarray,
    y: np.ndarray,
    model_registry: dict,
) -> dict[str, dict]:
    skf = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    all_predictions = {
        name: {"y_true": [], "y_pred": []} for name in model_registry
    }

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        X_train_selected, X_test_selected = select_k_best_features(
            X_train_fold, y_train_fold, X_test_fold
        )

        X_train_balanced, y_train_balanced = apply_smote(
            X_train_selected, y_train_fold
        )

        for model_name, model_factory in model_registry.items():
            model = model_factory()
            model.fit(X_train_balanced, y_train_balanced)
            fold_predictions = model.predict(X_test_selected).tolist()

            all_predictions[model_name]["y_true"].extend(y_test_fold.tolist())
            all_predictions[model_name]["y_pred"].extend(fold_predictions)

    results = {}
    for model_name, prediction_data in all_predictions.items():
        results[model_name] = compute_metrics(
            prediction_data["y_true"], prediction_data["y_pred"]
        )
    return results


def build_consolidated_dataframe(
    all_results: dict[str, dict[str, dict]]
) -> pd.DataFrame:
    rows = []
    for target_col, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            row = {
                "Variable Objetivo": target_col,
                "Modelo": model_name,
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)
