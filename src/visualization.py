import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import OUTPUTS_DIR


def plot_metric_comparison(
    consolidated_df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=consolidated_df,
        x="Variable Objetivo",
        y=metric,
        hue="Modelo",
        palette="viridis",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Variable Objetivo", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Gráfico guardado: {output_path}")


def plot_accuracy_comparison(consolidated_df: pd.DataFrame) -> None:
    plot_metric_comparison(
        consolidated_df,
        metric="Accuracy",
        title="Comparación de Accuracy por Modelo y Variable Objetivo",
        filename="accuracy_comparison.png",
    )


def plot_f1_comparison(consolidated_df: pd.DataFrame) -> None:
    plot_metric_comparison(
        consolidated_df,
        metric="F1 (macro)",
        title="Comparación de F1-Score (macro) por Modelo y Variable Objetivo",
        filename="f1_comparison.png",
    )


def generate_all_plots(consolidated_df: pd.DataFrame) -> None:
    print("\nGenerando gráficos comparativos...")
    plot_accuracy_comparison(consolidated_df)
    plot_f1_comparison(consolidated_df)
