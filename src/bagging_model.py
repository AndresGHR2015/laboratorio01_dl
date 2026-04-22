from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE

def create_bagging_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )
