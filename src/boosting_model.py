from sklearn.ensemble import GradientBoostingClassifier
from config import RANDOM_STATE

def create_boosting_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
