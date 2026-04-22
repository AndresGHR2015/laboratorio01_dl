import os
import pandas as pd

class DataFileNotFoundError(Exception):
    pass

def load_raw_data(filepath: str) -> pd.DataFrame:
    absolute_path = os.path.abspath(filepath)
    if not os.path.exists(absolute_path):
        raise DataFileNotFoundError(
            f"No se encontró el archivo de datos en: {absolute_path}"
        )
    return pd.read_spss(absolute_path)