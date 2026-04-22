import os

RANDOM_STATE = 42
K_BEST_FEATURES = 10
CV_SPLITS = 5
GDS_COLUMNS = ["GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]
COLUMNS_TO_EXCLUDE = ["ID"] + GDS_COLUMNS
DATA_FILEPATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "15 atributos R0-R5.sav"
)
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")