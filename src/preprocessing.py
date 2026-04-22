import os
import numpy as np
import pandas as pd
from config import COLUMNS_TO_EXCLUDE, PROCESSED_DATA_DIR

def prepare_data(
    df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = [
        col for col in df.columns if col not in COLUMNS_TO_EXCLUDE
    ]
    X = df[feature_columns].values
    y = df[target_col].values
    return X, y

def save_processed_data(
    df: pd.DataFrame,
    target_col: str,
    selected_feature_columns: list[str],
) -> str:
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_df = df[selected_feature_columns + [target_col]]
    output_filename = f"processed_{target_col}.csv"
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    processed_df.to_csv(output_path, index=False)
    return output_path