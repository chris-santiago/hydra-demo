"""Shared utilities for all demo versions."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# Constants
FEATURE_COLS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
CAT_FEATURES = ["sex", "embarked"]
NUM_FEATURES = ["age", "fare"]
TARGET_COL = "survived"


def load_titanic(test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    """
    Load Titanic dataset from OpenML and prepare train/test split.

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split by target variable

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pandas DataFrames/Series
    """
    # Fetch Titanic dataset
    titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = titanic.frame

    # Select features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Convert target to numeric (it comes as categorical from OpenML)
    y = y.astype(int)

    # Convert categorical columns to string type
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model and return metrics.

    Args:
        model: Trained sklearn model (or pipeline)
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with accuracy, f1_score, and classification_report
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(y_test, y_pred),
    }

    return metrics


def display_config(cfg: Any, title: str = "Configuration"):
    """
    Pretty-print a configuration object.

    Args:
        cfg: Configuration (dict, DictConfig, or OmegaConf object)
        title: Title to display above the config
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print("=" * 60)

    if isinstance(cfg, (DictConfig, dict)):
        if isinstance(cfg, DictConfig):
            print(OmegaConf.to_yaml(cfg))
        else:
            import yaml

            print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    else:
        print(cfg)

    print("=" * 60 + "\n")


def save_artifacts(
    output_dir: str | Path,
    pipeline,
    metrics: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Save model artifacts to the output directory.

    Args:
        output_dir: Directory to save artifacts
        pipeline: Trained sklearn pipeline
        metrics: Dictionary with evaluation metrics
        X_test: Test features (for feature importance)
        y_test: Test labels (for classification report)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
            },
            f,
            indent=2,
        )
    print(f"✓ Saved metrics to {metrics_file}")

    # Save full classification report as text
    report_file = output_path / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(metrics["classification_report"])
    print(f"✓ Saved classification report to {report_file}")

    # Try to save encoder mapping (if using OrdinalEncoder or OneHotEncoder)
    try:
        # Access the encoder from the pipeline (assumed to be step 2)
        if hasattr(pipeline, "named_steps"):
            encoder = pipeline.named_steps.get("encoder")
            if encoder and hasattr(encoder, "encoder_dict_"):
                mapping_file = output_path / "encoder_mapping.json"
                # Convert numpy types to native Python types for JSON serialization
                encoder_dict = {
                    k: {str(key): int(val) for key, val in v.items()}
                    if isinstance(v, dict)
                    else str(v)
                    for k, v in encoder.encoder_dict_.items()
                }
                with open(mapping_file, "w") as f:
                    json.dump(encoder_dict, f, indent=2)
                print(f"✓ Saved encoder mapping to {mapping_file}")
    except Exception as e:
        print(f"  (Could not save encoder mapping: {e})")

    # Try to save feature importance (if model supports it)
    try:
        if hasattr(pipeline, "named_steps"):
            model = pipeline.named_steps.get("classifier")
            if model and hasattr(model, "feature_importances_"):
                # Get feature names after encoding
                transformed = pipeline.named_steps["encoder"].transform(X_test)
                feature_names = transformed.columns.tolist()

                importance_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                importance_file = output_path / "feature_importance.csv"
                importance_df.to_csv(importance_file, index=False)
                print(f"✓ Saved feature importance to {importance_file}")
    except Exception as e:
        print(f"  (Could not save feature importance: {e})")
