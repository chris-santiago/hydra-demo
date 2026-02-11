"""
Step 6: Advanced Hydra Features

Demonstrates:
1. --multirun: Sweep over multiple configs in parallel
   uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting

2. Experiment presets: Pre-configured combinations
   uv run python src/v6_hydra_advanced.py +experiment=quick_test

3. Compose API: Programmatic config for notebooks/testing
   Can be imported and used: compose_and_run("random_forest", "mean")

4. Custom output directory naming:
   Organized by model type and timestamp for easy navigation
"""

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
from sklearn.pipeline import Pipeline

from utils import display_config, evaluate_model, load_titanic, save_artifacts


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Train models with advanced Hydra features."""
    print("\n" + "=" * 60)
    print("STEP 6: Advanced Hydra Features".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration")

    # Load data
    print("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_state,
    )

    # Instantiate all components
    print(f"Instantiating: {cfg.model._target_}...")
    encoder = hydra.utils.instantiate(cfg.preprocessing)
    classifier = hydra.utils.instantiate(cfg.model)
    num_imputer = hydra.utils.instantiate(cfg.num_imputer)
    cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

    # Build and train pipeline
    print("Building and training pipeline...")
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
    print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ F1 Score: {metrics['f1_score']:.4f}")

    # Save artifacts
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"\nSaving artifacts to: {output_dir}")
    save_artifacts(output_dir, pipeline, metrics, X_test, y_test)

    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATED:")
    print("✓ --multirun for parameter sweeps")
    print("✓ +experiment= for preset configs")
    print("✓ Custom output dir structure")
    print("✓ Compose API (see compose_and_run below)")
    print("=" * 60 + "\n")

    return metrics["accuracy"]  # Can be used by optimization frameworks


def compose_and_run(model_name: str = "random_forest", preprocessing: str = "ordinal"):
    """
    Compose API demo: Programmatically create and run config.

    Useful for:
    - Jupyter notebooks
    - Testing
    - Integration with other tools
    - Programmatic experimentation

    Example:
        from v6_hydra_advanced import compose_and_run
        acc = compose_and_run("gradient_boosting", "mean")
    """
    # Get absolute path to config directory
    config_dir = str(Path(__file__).parent.parent / "conf")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Compose config programmatically
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                f"preprocessing={preprocessing}",
            ],
        )

        print(f"\n{'=' * 60}")
        print(
            f"Compose API: model={model_name}, preprocessing={preprocessing}".center(60)
        )
        print("=" * 60)

        # Load data
        X_train, X_test, y_train, y_test = load_titanic(
            test_size=cfg.dataset.test_size,
            random_state=cfg.random_state,
        )

        # Instantiate and train
        encoder = hydra.utils.instantiate(cfg.preprocessing)
        classifier = hydra.utils.instantiate(cfg.model)
        num_imputer = hydra.utils.instantiate(cfg.num_imputer)
        cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

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
        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ F1 Score: {metrics['f1_score']:.4f}\n")

        return metrics["accuracy"]


if __name__ == "__main__":
    main()
