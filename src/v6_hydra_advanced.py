"""
Step 6: Advanced Hydra Features

Demonstrates:
1. --multirun: Sweep over multiple configs sequentially
   uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting

2. Experiment presets: Pre-configured combinations
   uv run python src/v6_hydra_advanced.py +experiment=quick_test

3. Custom output directory naming:
   Organized by model type and timestamp for easy navigation
"""

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic, save_artifacts

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Train models with advanced Hydra features."""
    print("\n" + "=" * 60)
    print("STEP 6: Advanced Hydra Features".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration")

    # Load data
    log.info("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_state,
        stratify=cfg.dataset.stratify,
    )

    # Instantiate all components
    log.info(f"Instantiating: {cfg.model._target_}...")
    encoder = hydra.utils.instantiate(cfg.preprocessing)
    classifier = hydra.utils.instantiate(cfg.model)
    num_imputer = hydra.utils.instantiate(cfg.num_imputer)
    cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

    # Build and train pipeline
    log.info("Building and training pipeline...")
    pipeline = Pipeline(
        [
            ("num_imputer", num_imputer),
            ("cat_imputer", cat_imputer),
            ("encoder", encoder),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    log.info(f"Accuracy: {metrics['accuracy']:.4f}")
    log.info(f"F1 Score: {metrics['f1_score']:.4f}")

    # Save artifacts
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving artifacts to: {output_dir}")
    save_artifacts(output_dir, pipeline, metrics, X_test, y_test)

    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATED:")
    print("✓ --multirun for parameter sweeps")
    print("✓ +experiment= for preset configs")
    print("✓ Custom output dir structure")
    print("=" * 60 + "\n")

    return metrics["accuracy"]  # Can be used by optimization frameworks


if __name__ == "__main__":
    main()
