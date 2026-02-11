"""
Step 1: Pure Python - Everything Hardcoded

Pain points:
- All parameters hardcoded in source
- To try a different model, you must edit the code
- No configuration management
- No experiment tracking
"""

from feature_engine.encoding import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic


def main():
    """Train a hardcoded logistic regression model."""
    print("\n" + "=" * 60)
    print("STEP 1: Pure Python (Hardcoded)".center(60))
    print("=" * 60)

    # Hardcoded parameters
    config = {
        "model": "LogisticRegression",
        "encoding": "OrdinalEncoder",
        "test_size": 0.2,
        "random_state": 42,
        "max_iter": 1000,
    }

    display_config(config, "Hardcoded Configuration")

    # Load data
    print("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    # Build pipeline - everything hardcoded!
    print("Building pipeline with hardcoded components...")
    pipeline = Pipeline(
        [
            (
                "num_imputer",
                MeanMedianImputer(imputation_method="mean", variables=["age", "fare"]),
            ),
            (
                "cat_imputer",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=["sex", "embarked"],
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    variables=["sex", "embarked"],
                    ignore_format=True,
                    missing_values="ignore",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config["max_iter"],
                    random_state=config["random_state"],
                ),
            ),
        ]
    )

    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ F1 Score: {metrics['f1_score']:.4f}")
    print(f"\n{metrics['classification_report']}")

    print("\n" + "=" * 60)
    print("PAIN POINT: Want to try RandomForest? Edit the source code!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
