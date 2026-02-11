"""
Step 3: Basic Hydra - Structured Configuration

Improvements:
- Structured configuration (YAML + dot notation)
- Auto-saved config in outputs/ directory
- Cleaner CLI: cfg.model.name instead of --model
- Type-safe access with OmegaConf

Pain points:
- Still has if/elif chains for dispatch
- Flat config (no config groups yet)
"""

import hydra
from feature_engine.encoding import MeanEncoder, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic


@hydra.main(version_base=None, config_path="../conf", config_name="config_v3")
def main(cfg: DictConfig):
    """Train a model with basic Hydra configuration."""
    print("\n" + "=" * 60)
    print("STEP 3: Basic Hydra".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration")

    # Load data
    print("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    # Still need if/elif for encoder selection
    print("Selecting encoder...")
    if cfg.preprocessing.encoding == "ordinal":
        encoder = OrdinalEncoder(
            variables=["sex", "embarked"],
            ignore_format=True,
            missing_values="ignore",
        )
    elif cfg.preprocessing.encoding == "onehot":
        encoder = OneHotEncoder(
            variables=["sex", "embarked"],
            drop_last=False,
        )
    elif cfg.preprocessing.encoding == "mean":
        encoder = MeanEncoder(
            variables=["sex", "embarked"],
            ignore_format=True,
            missing_values="ignore",
        )
    else:
        raise ValueError(f"Unknown encoding: {cfg.preprocessing.encoding}")

    # Still need if/elif for model selection
    print("Selecting model...")
    if cfg.model.name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=cfg.model.max_iter,
            random_state=cfg.random_state,
        )
    elif cfg.model.name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.random_state,
        )
    elif cfg.model.name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

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
    print("IMPROVEMENTS:")
    print("✓ Structured config with dot notation")
    print("✓ Auto-saved in outputs/YYYY-MM-DD/HH-MM-SS/.hydra/")
    print("\nPAIN POINT:")
    print("- Still need if/elif chains for dispatch")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
