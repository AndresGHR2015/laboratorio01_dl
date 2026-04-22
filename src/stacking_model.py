from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from bagging_model import create_bagging_model
from boosting_model import create_boosting_model


def create_stacking_model() -> StackingClassifier:
    base_estimators = [
        ("random_forest", create_bagging_model()),
        ("gradient_boosting", create_boosting_model()),
    ]
    final_estimator = LogisticRegression(max_iter=1000)
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
    )
