import warnings
import pandas as pd
from bagging_model import create_bagging_model
from boosting_model import create_boosting_model
from config import COLUMNS_TO_EXCLUDE, DATA_FILEPATH, GDS_COLUMNS
from data_loader import load_raw_data
from evaluation import build_consolidated_dataframe, evaluate_models_for_target
from preprocessing import prepare_data, save_processed_data
from stacking_model import create_stacking_model
from visualization import generate_all_plots

warnings.filterwarnings("ignore")

MODEL_REGISTRY = {
    "Bagging (Random Forest)": create_bagging_model,
    "Boosting (Gradient Boosting)": create_boosting_model,
    "Stacking": create_stacking_model,
}


def print_header():
    print("=" * 70)
    print("  Laboratorio 01: Clasificación de Deterioro Cognitivo")
    print("  Modelos: Bagging | Boosting | Stacking")
    print("  Validación: StratifiedKFold (k=5) + SMOTE en Train")
    print("  Selección: Chi-Cuadrado (top-10 atributos)")
    print("=" * 70)


def print_target_results(target_col: str, results: dict[str, dict]):
    print(f"\n{'─' * 70}")
    print(f"  Variable Objetivo: {target_col}")
    print(f"{'─' * 70}")
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Modelo"
    print(results_df.to_markdown(floatfmt=".4f"))


def print_consolidated_tables(
    consolidated_df: pd.DataFrame, model_names: list[str]
):
    print(f"\n{'=' * 70}")
    print("  TABLA CONSOLIDADA (para copiar al informe)")
    print(f"{'=' * 70}\n")

    pivot_accuracy = consolidated_df.pivot(
        index="Variable Objetivo", columns="Modelo", values="Accuracy"
    )
    pivot_accuracy = pivot_accuracy[
        [name for name in model_names if name in pivot_accuracy.columns]
    ]
    print("### Accuracy por Modelo y Variable Objetivo\n")
    print(pivot_accuracy.to_markdown(floatfmt=".4f"))

    print()

    pivot_f1 = consolidated_df.pivot(
        index="Variable Objetivo", columns="Modelo", values="F1 (macro)"
    )
    pivot_f1 = pivot_f1[
        [name for name in model_names if name in pivot_f1.columns]
    ]
    print("### F1-Score (macro) por Modelo y Variable Objetivo\n")
    print(pivot_f1.to_markdown(floatfmt=".4f"))
    print()


def main():
    print_header()

    df = load_raw_data(DATA_FILEPATH)
    print(
        f"\n✔ Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas"
    )

    all_results = {}

    feature_columns = [col for col in df.columns if col not in COLUMNS_TO_EXCLUDE]

    for target_col in GDS_COLUMNS:
        print(f"\n▶ Evaluando variable objetivo: {target_col} ...")
        X, y = prepare_data(df, target_col)
        saved_path = save_processed_data(df, target_col, feature_columns)
        print(f"  ✔ Dataset procesado guardado en: {saved_path}")
        model_results = evaluate_models_for_target(X, y, MODEL_REGISTRY)
        all_results[target_col] = model_results
        print_target_results(target_col, model_results)

    consolidated_df = build_consolidated_dataframe(all_results)

    print_consolidated_tables(consolidated_df, list(MODEL_REGISTRY.keys()))

    generate_all_plots(consolidated_df)


if __name__ == "__main__":
    main()
