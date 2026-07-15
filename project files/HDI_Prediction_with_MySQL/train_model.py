"""Train, evaluate, visualize, and serialize the HDI linear regression model.

Run this file from the HDI_Prediction directory:
    python train_model.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "HumanDevelopmentIndex.csv"
PLOT_DIR = BASE_DIR / "static" / "plots"
MODEL_PATH = BASE_DIR / "hdi_model.pkl"
FEATURES = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "Gross National Income per Capita",
]
TARGET = "HDI"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Load the CSV and replace missing numeric values with each column mean."""
    data = pd.read_csv(path)
    print("\nFirst five rows:\n", data.head())
    print("\nDataset information:")
    data.info()
    print("\nSummary statistics:\n", data.describe(include="all"))
    print("\nDataset shape:", data.shape)
    print("\nCountries:\n", data["Country"].unique())
    print("\nMissing values before filling:\n", data.isna().sum())

    numeric_columns = data.select_dtypes(include="number").columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    print("\nMissing values after filling:\n", data.isna().sum())
    return data


def create_visualizations(data: pd.DataFrame) -> None:
    """Create the requested exploratory data-analysis plots as PNG files."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    plt.figure(figsize=(11, 5))
    sns.stripplot(data=data, x="Development Group", y=TARGET, jitter=0.18, size=6)
    plt.title("HDI Distribution by Development Group")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "strip_plot_hdi.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(data=data, x="Life Expectancy", y=TARGET, scatter_kws={"s": 55})
    plt.title("Life Expectancy vs HDI")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "life_expectancy_vs_hdi.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.regplot(data=data, x="Mean Years of Schooling", y=TARGET, scatter_kws={"s": 55})
    plt.title("Mean Years of Schooling vs HDI")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "schooling_vs_hdi.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    correlation = data[FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap="YlGnBu", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def train_and_save(data: pd.DataFrame) -> float:
    """Split data, train a model, print evaluation, and write the pickle artifact."""
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    score = r2_score(y_test, predictions)

    comparison = pd.DataFrame(
        {"Actual HDI": y_test.to_numpy(), "Predicted HDI": predictions}
    ).round(4)
    print(f"\nR² score: {score:.4f}")
    print("\nActual vs predicted values:\n", comparison.to_string(index=False))

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)
    return score


def main() -> None:
    """Execute the full model-training workflow."""
    data = load_and_prepare_data(DATA_PATH)
    create_visualizations(data)
    score = train_and_save(data)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Final R² score: {score:.4f}")


if __name__ == "__main__":
    main()
