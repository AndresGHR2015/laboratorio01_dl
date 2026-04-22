import numpy as np
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE

MINIMUM_SAMPLES_FOR_SMOTE = 2

def apply_smote(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()

    if min_class_count < MINIMUM_SAMPLES_FOR_SMOTE:
        return X, y

    safe_k_neighbors = min(min_class_count - 1, 5)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=safe_k_neighbors)
    return smote.fit_resample(X, y)
