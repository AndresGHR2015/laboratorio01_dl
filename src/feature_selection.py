import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from config import K_BEST_FEATURES

class FeatureSelectionError(Exception):
    pass


def select_k_best_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = K_BEST_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    available_features = X_train.shape[1]
    if k > available_features:
        raise FeatureSelectionError(
            f"Se solicitaron {k} características pero solo hay {available_features} disponibles."
        )

    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected