"""
Step 4: Hydra Config Groups - Composable Configuration

Improvements:
- Config groups: model=random_forest swaps entire config
- Each model file carries only its own parameters (no param leaking)
- Swappable preprocessing strategies
- Uses _target_ for dispatch (dict registry lookup)

Pain points:
- Still need manual registry dicts for model/encoder classes
- Still importing all classes at top of file
- Manual instantiation with **cfg boilerplate
"""

import logging

import hydra
from feature_engine.encoding import MeanEncoder, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic

log = logging.getLogger(__name__)

# Manual registries - still need to import everything!
ENCODER_REGISTRY = {
    "feature_engine.encoding.OrdinalEncoder": OrdinalEncoder,
    "feature_engine.encoding.OneHotEncoder": OneHotEncoder,
    "feature_engine.encoding.MeanEncoder": MeanEncoder,
}

MODEL_REGISTRY = {
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    "sklearn.ensemble.GradientBoostingClassifier": GradientBoostingClassifier,
}


def build_from_config(cfg: DictConfig, registry: dict):
    """
    Build an object from config using a registry.

    Reads _target_ key and looks up class in registry, then
    instantiates with remaining config keys as kwargs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    target = cfg_dict.pop("_target_")
    # Remove Hydra-specific keys that shouldn't be passed to constructors
    cfg_dict.pop("_convert_", None)
    cfg_dict.pop("_recursive_", None)
    cls = registry[target]
    return cls(**cfg_dict)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Train a model with Hydra config groups."""
    print("\n" + "=" * 60)
    print("STEP 4: Hydra Config Groups".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration with Groups")

    # Load data
    log.info("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_state,
        stratify=cfg.dataset.stratify,
    )

    # Build encoder using registry (no more if/elif!)
    log.info(f"Building encoder: {cfg.preprocessing._target_}...")
    encoder = build_from_config(cfg.preprocessing, ENCODER_REGISTRY)

    # Build model using registry
    log.info(f"Building model: {cfg.model._target_}...")
    classifier = build_from_config(cfg.model, MODEL_REGISTRY)

    # Build imputers using registry
    num_imputer = MeanMedianImputer(imputation_method="mean", variables=["age", "fare"])
    cat_imputer = CategoricalImputer(
        imputation_method="frequent",
        variables=["sex", "embarked"],
    )

    # Build pipeline
    log.info("Building pipeline...")
    pipeline = Pipeline(
        [
            ("num_imputer", num_imputer),
            ("cat_imputer", cat_imputer),
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
    print("✓ Config groups: model=random_forest swaps everything")
    print("✓ No parameter leaking (each YAML has only relevant params)")
    print("✓ No if/elif chains!")
    print("\nPAIN POINT:")
    print("- Still need manual registries + imports")
    print("- Manual build_from_config() boilerplate")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
