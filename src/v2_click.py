"""
Step 2: Click CLI - Command-Line Arguments

Improvements:
- CLI arguments for flexibility
- Can change model without editing code

Pain points:
- if/elif chains for model/encoder selection
- All possible parameters exposed as options (even irrelevant ones)
- max_iter shows for RandomForest, n_estimators shows for LogisticRegression
- Adding new model requires editing decorators AND if/elif in 2 places
- No automatic config saving or experiment tracking
"""

import click
from feature_engine.encoding import MeanEncoder, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic


@click.command()
@click.option(
    "--model",
    type=click.Choice(["logistic_regression", "random_forest", "gradient_boosting"]),
    default="logistic_regression",
    help="Model to train",
)
@click.option(
    "--encoding",
    type=click.Choice(["ordinal", "onehot", "mean"]),
    default="ordinal",
    help="Categorical encoding method",
)
@click.option("--test-size", type=float, default=0.2, help="Test set size")
@click.option("--random-state", type=int, default=42, help="Random seed")
@click.option(
    "--max-iter", type=int, default=1000, help="Max iterations (LogisticRegression)"
)
@click.option(
    "--n-estimators",
    type=int,
    default=100,
    help="Number of estimators (RF/GBM)",
)
@click.option("--max-depth", type=int, default=None, help="Max tree depth (RF/GBM)")
@click.option("--learning-rate", type=float, default=0.1, help="Learning rate (GBM)")
def main(
    model,
    encoding,
    test_size,
    random_state,
    max_iter,
    n_estimators,
    max_depth,
    learning_rate,
):
    """Train a model with Click CLI arguments."""
    print("\n" + "=" * 60)
    print("STEP 2: Click CLI".center(60))
    print("=" * 60)

    config = {
        "model": model,
        "encoding": encoding,
        "test_size": test_size,
        "random_state": random_state,
        "max_iter": max_iter,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }

    display_config(config, "Click Configuration")

    # Load data
    print("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=test_size,
        random_state=random_state,
    )

    # Manual if/elif for encoder selection
    print("Selecting encoder...")
    if encoding == "ordinal":
        encoder = OrdinalEncoder(
            variables=["sex", "embarked"],
            ignore_format=True,
            missing_values="ignore",
        )
    elif encoding == "onehot":
        encoder = OneHotEncoder(
            variables=["sex", "embarked"],
            drop_last=False,
        )
    elif encoding == "mean":
        encoder = MeanEncoder(
            variables=["sex", "embarked"],
            ignore_format=True,
            missing_values="ignore",
        )
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Manual if/elif for model selection
    print("Selecting model...")
    if model == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    elif model == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Build pipeline
    print("Building pipeline...")
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
            ("encoder", encoder),
            ("classifier", classifier),
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
    print("PAIN POINTS:")
    print("- if/elif chains in 2 places (encoder + model)")
    print("- All params exposed (--max-iter visible for RF)")
    print("- Want warm_start? Add decorator + thread through if/elif")
    print("- No config saving or experiment tracking")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
