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

import logging

import hydra
from feature_engine.encoding import MeanEncoder, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config_v3")
def main(cfg: DictConfig):
    """Train a model with basic Hydra configuration."""
    print("\n" + "=" * 60)
    print("STEP 3: Basic Hydra".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration")

    # Load data
    log.info("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        # config_v3.yaml doesn't have dataset.stratify, so uses default (True)
    )

    # Still need if/elif for encoder selection
    log.info("Selecting encoder...")
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
    log.info("Selecting model...")
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
    log.info("Building pipeline...")
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
    log.info("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    log.info(f"Accuracy: {metrics['accuracy']:.4f}")
    log.info(f"F1 Score: {metrics['f1_score']:.4f}")
    log.info(f"\n{metrics['classification_report']}")

    print("\n" + "=" * 60)
    print("IMPROVEMENTS:")
    print("✓ Structured config with dot notation")
    print("✓ Auto-saved in outputs/YYYY-MM-DD/HH-MM-SS/.hydra/")
    print("\nPAIN POINT:")
    print("- Still need if/elif chains for dispatch")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
