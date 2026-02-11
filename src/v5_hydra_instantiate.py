"""
Step 5: Hydra instantiate() - The "Aha!" Moment

THE KEY BREAKTHROUGH:
- hydra.utils.instantiate(cfg.model) replaces ALL registries and if/elif
- Training script is ~15 lines, never needs to change
- Adding a new model = creating one YAML file
- Zero boilerplate, config IS the code

DYNAMIC PARAMETER ADDITION (+ syntax):
This is where Hydra truly shines vs Click/argparse:

Click requires:
1. Add @click.option decorator
2. Add param to function signature
3. Thread through if/elif
4. Redeploy

Hydra allows:
  uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true

The + prefix adds a key not in the YAML. Since instantiate() passes all keys as kwargs,
warm_start=True goes straight to RandomForestClassifier(warm_start=True, ...).
NO CODE CHANGES NEEDED!

Examples:
  # Add penalty and solver to logistic regression
  uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

  # Override existing param (no + needed)
  uv run python src/v5_hydra_instantiate.py model=random_forest model.n_estimators=500

  # Combine: swap + override + add new
  uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.n_estimators=200 +model.subsample=0.8
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
    """Train a model with Hydra instantiate() - zero boilerplate!"""
    print("\n" + "=" * 60)
    print("STEP 5: Hydra instantiate() - The Magic!".center(60))
    print("=" * 60)

    display_config(cfg, "Hydra Configuration")

    # Load data
    log.info("Loading Titanic dataset...")
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_state,
        stratify=cfg.dataset.stratify,
    )

    # Build components with instantiate() - NO imports, NO registries!
    log.info(f"Instantiating encoder: {cfg.preprocessing._target_}...")
    encoder = hydra.utils.instantiate(cfg.preprocessing)

    log.info(f"Instantiating model: {cfg.model._target_}...")
    classifier = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating imputers...")
    num_imputer = hydra.utils.instantiate(cfg.num_imputer)
    cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

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

    # Save artifacts to Hydra's output directory
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving artifacts to: {output_dir}")
    save_artifacts(output_dir, pipeline, metrics, X_test, y_test)

    print("\n" + "=" * 60)
    print("THE AHA! MOMENT:")
    print("✓ instantiate() replaced ALL boilerplate")
    print("✓ Training script is ~15 lines")
    print("✓ Adding new model = one YAML file")
    print("✓ Dynamic params via +: +model.warm_start=true")
    print("✓ Auto-saved config + custom artifacts")
    print("✓ Config IS the code!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
