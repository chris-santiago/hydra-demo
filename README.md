# Hydra Teaching Demo: From Chaos to Clarity

A step-by-step demonstration of Hydra's core concepts for ML experiment management. This project progressively evolves a training script from hardcoded pure Python through Click CLI to full Hydra — showing the **"aha moment"** when `instantiate()` eliminates all boilerplate.

## What You'll Learn

1. **The Problem**: Why hardcoded configs and CLI tools become painful
2. **Config Groups**: How to compose swappable configurations
3. **The Magic**: How `instantiate()` eliminates all boilerplate
4. **Dynamic Parameters**: Adding new params without code changes (`+` syntax)
5. **Experiment Tracking**: Automatic config saving and artifact management
6. **Advanced Features**: Multirun sweeps, experiment presets, Compose API

## Setup

```bash
# Install dependencies (uses uv for fast, reproducible installs)
uv sync
```

## The Journey: Six Progressive Steps

### Step 1: Pure Python (Hardcoded) — The Pain

**File**: `src/v1_pure_python.py`

Everything is hardcoded. Want to try RandomForest? **Edit the source code.**

```bash
uv run python src/v1_pure_python.py
```

**Pain Points**:
- ❌ All parameters baked into source
- ❌ Changing models requires code edits
- ❌ No configuration management
- ❌ No experiment tracking

---

### Step 2: Click CLI — Partial Relief

**File**: `src/v2_click.py`

CLI arguments provide flexibility, but at a cost.

```bash
# Basic usage
uv run python src/v2_click.py --model random_forest --encoding onehot

# With parameters
uv run python src/v2_click.py --model gradient_boosting --n-estimators 200 --learning-rate 0.05
```

**Improvements**:
- ✅ Can change model via CLI

**Pain Points**:
- ❌ `if/elif` chains in 2 places (encoder + model selection)
- ❌ All params exposed (e.g., `--max-iter` visible for RandomForest)
- ❌ Want `warm_start`? Add `@click.option` decorator + thread through `if/elif` + redeploy
- ❌ No automatic config saving

**The "Add a Parameter" Problem**:
To add `warm_start=True` to RandomForest:
1. Add `@click.option("--warm-start", ...)` decorator
2. Add `warm_start` to function signature
3. Thread it into the `if/elif` branch for RandomForest
4. Redeploy

This is what Hydra solves elegantly (see v5).

---

### Step 3: Basic Hydra — Structured Config

**File**: `src/v3_hydra_basic.py`
**Config**: `conf/config_v3.yaml`

Hydra introduces structured configuration with dot-notation CLI overrides.

```bash
# Override config values with dot notation
uv run python src/v3_hydra_basic.py model.name=random_forest

# Multiple overrides
uv run python src/v3_hydra_basic.py model.name=gradient_boosting model.n_estimators=200 test_size=0.3
```

**Improvements**:
- ✅ Structured YAML configuration
- ✅ Auto-saved config in `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`
- ✅ Cleaner CLI with dot notation

**Pain Points**:
- ❌ Still has `if/elif` chains for dispatch
- ❌ Flat config structure (no composition yet)

---

### Step 4: Config Groups — Swappable Components

**File**: `src/v4_hydra_groups.py`
**Config**: `conf/config.yaml` + `conf/model/*.yaml` + `conf/preprocessing/*.yaml`

Config groups allow you to **swap entire configurations** with a single argument.

```bash
# Swap model (loads conf/model/random_forest.yaml)
uv run python src/v4_hydra_groups.py model=random_forest

# Swap both model and preprocessing
uv run python src/v4_hydra_groups.py model=gradient_boosting preprocessing=mean

# Override parameters within a group
uv run python src/v4_hydra_groups.py model=random_forest model.n_estimators=500
```

**How It Works**:
- `model=random_forest` loads `conf/model/random_forest.yaml`
- Each model YAML has only its own parameters (no leaking!)
- `_target_: sklearn.ensemble.RandomForestClassifier` specifies the class

**Improvements**:
- ✅ Component swapping: `model=random_forest` loads entire config
- ✅ No parameter leaking (each YAML has only relevant params)
- ✅ No `if/elif` chains!

**Pain Points**:
- ❌ Still need manual registry dicts
- ❌ Still importing all model/encoder classes
- ❌ Manual `build_from_config()` boilerplate

---

### Step 5: `instantiate()` — The "Aha!" Moment ✨

**File**: `src/v5_hydra_instantiate.py`
**Config**: Same as v4

This is where Hydra **truly shines**. `hydra.utils.instantiate()` eliminates ALL boilerplate.

```bash
# Basic usage
uv run python src/v5_hydra_instantiate.py model=random_forest preprocessing=mean

# Override existing parameters
uv run python src/v5_hydra_instantiate.py model=random_forest model.n_estimators=500
```

**The Training Script** (core logic):
```python
# That's it! No imports, no registries, no if/elif!
encoder = hydra.utils.instantiate(cfg.preprocessing)
classifier = hydra.utils.instantiate(cfg.model)
```

**The Config** (`conf/model/random_forest.yaml`):
```yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
max_depth: null
random_state: ${random_state}
```

**What `instantiate()` Does**:
1. Reads `_target_: sklearn.ensemble.RandomForestClassifier`
2. Imports the class dynamically
3. Passes remaining keys as `**kwargs`
4. Returns `RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)`

**Adding a New Model**: Just create `conf/model/my_model.yaml`:
```yaml
_target_: sklearn.svm.SVC
C: 1.0
kernel: rbf
random_state: ${random_state}
```

Then: `uv run python src/v5_hydra_instantiate.py model=my_model`

**No code changes needed!**

#### 🎯 Dynamic Parameter Addition: The `+` Syntax

This is a game-changer compared to Click/argparse.

**The Problem (Click)**:
Want to pass `warm_start=True` to RandomForest?
1. Add `@click.option("--warm-start", ...)` decorator
2. Add `warm_start` to function signature
3. Thread it through `if/elif` branch
4. Redeploy

**The Solution (Hydra)**:
```bash
# Add a parameter NOT in the YAML (+ prefix required)
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
```

The `+` tells Hydra: "Add this key even though it's not in `conf/model/random_forest.yaml`."
Since `instantiate()` passes all keys as `**kwargs`, it goes straight to `RandomForestClassifier(warm_start=True, ...)`.

**More Examples**:
```bash
# Add penalty and solver to logistic regression
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Override existing param (no + needed)
uv run python src/v5_hydra_instantiate.py model=random_forest model.n_estimators=500

# Combine: swap + override existing + add new
uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.n_estimators=200 +model.subsample=0.8
```

**Why This Matters**:
| Action | Click/Argparse | Hydra |
|--------|----------------|-------|
| Add new param | Edit decorators + code + redeploy | `+model.param=value` |
| Override param | `--param value` | `model.param=value` |
| Code changes | Required | **Zero** |

#### 📦 Experiment Tracking & Artifacts

Hydra automatically creates timestamped output directories with all experiment metadata.

**What Hydra Auto-Saves** (in `.hydra/` subdirectory):
- `config.yaml` — The fully resolved config (exactly what `cfg` contains)
- `overrides.yaml` — Only the CLI overrides used for this run
- `hydra.yaml` — Full Hydra runtime metadata

**Custom Artifacts We Save** (in v5/v6):
- `metrics.json` — accuracy, F1 score
- `classification_report.txt` — full sklearn classification report
- `encoder_mapping.json` — fitted encoder's mapping (e.g., ordinal: `{"male": 0, "female": 1}`)
- `feature_importance.csv` — if model supports `feature_importances_` (RF, GBM)

**Demo**:
```bash
# Run an experiment
uv run python src/v5_hydra_instantiate.py model=random_forest

# Inspect what was saved
ls outputs/$(ls -t outputs | head -1)/

# Should show:
# .hydra/config.yaml          # Full resolved config
# .hydra/overrides.yaml        # What you passed on CLI
# .hydra/hydra.yaml            # Hydra metadata
# metrics.json                 # Your custom metrics
# classification_report.txt    # Detailed evaluation
# encoder_mapping.json         # Encoder state
# feature_importance.csv       # Feature importances

# View the exact config used
cat outputs/$(ls -t outputs | head -1)/.hydra/config.yaml

# Reproduce the run
cat outputs/$(ls -t outputs | head -1)/.hydra/overrides.yaml
# Copy those overrides and re-run!
```

**The Reproducibility Story**:
| | v1/v2 (Pure Python/Click) | v5/v6 (Hydra) |
|---|---|---|
| Config saved? | ❌ Lost after run | ✅ Auto-saved in `.hydra/config.yaml` |
| Overrides tracked? | ❌ No | ✅ Auto-saved in `.hydra/overrides.yaml` |
| Artifacts organized? | ❌ Dumped in project root | ✅ Per-experiment output dirs |
| Reproduce a run? | ❌ Hope you remember CLI args | ✅ `cat .hydra/overrides.yaml` and re-run |

**Improvements**:
- ✅ `instantiate()` replaced ALL boilerplate
- ✅ Training script is ~15 lines (never needs to change!)
- ✅ Adding new model = one YAML file
- ✅ Dynamic params via `+model.param=value` (no code changes!)
- ✅ Auto-saved config + custom artifacts
- ✅ **Config IS the code**

---

### Step 6: Advanced Features — Production Ready

**File**: `src/v6_hydra_advanced.py`
**Config**: Same as v5 + `conf/experiment/*.yaml`

#### 1. Multirun Sweeps

Run multiple experiments in parallel:

```bash
# Sweep over 3 models
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting

# Sweep over model x preprocessing (3 x 3 = 9 runs)
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting preprocessing=ordinal,onehot,mean
```

Each run gets its own output directory under `multirun/YYYY-MM-DD_HH-MM-SS/`.

#### 2. Experiment Presets

Pre-configured combinations for common scenarios:

```bash
# Quick test: logistic regression, larger test set
uv run python src/v6_hydra_advanced.py +experiment=quick_test

# Full comparison: random forest with more trees
uv run python src/v6_hydra_advanced.py +experiment=full_comparison
```

**Experiment Config** (`conf/experiment/quick_test.yaml`):
```yaml
# @package _global_
defaults:
  - override /model: logistic_regression
  - override /preprocessing: ordinal

dataset:
  test_size: 0.3

model:
  max_iter: 500
```

#### 3. Compose API (Programmatic Config)

For Jupyter notebooks, testing, or integration with other tools:

```python
from src.v6_hydra_advanced import compose_and_run

# Programmatically run experiments
accuracy = compose_and_run("gradient_boosting", "mean")
```

#### 4. Custom Output Directory Structure

Organize outputs by model type and timestamp:

**Config** (`conf/config.yaml`):
```yaml
hydra:
  run:
    dir: outputs/${model._target_}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.override_dirname}
```

**Result**:
```
outputs/
├── sklearn.ensemble.RandomForestClassifier/
│   ├── 2026-02-10_14-30-22/
│   │   ├── .hydra/
│   │   ├── metrics.json
│   │   └── ...
│   └── 2026-02-10_15-45-10/
└── sklearn.linear_model.LogisticRegression/
    └── 2026-02-10_14-35-55/
```

---

## Complete Command Reference

### Basic Operations

```bash
# v1: Hardcoded
uv run python src/v1_pure_python.py

# v2: Click CLI
uv run python src/v2_click.py --model random_forest --encoding onehot
uv run python src/v2_click.py --model gradient_boosting --n-estimators 200

# v3: Basic Hydra
uv run python src/v3_hydra_basic.py model.name=random_forest
uv run python src/v3_hydra_basic.py model.name=gradient_boosting test_size=0.3

# v4: Config Groups
uv run python src/v4_hydra_groups.py model=random_forest preprocessing=mean
uv run python src/v4_hydra_groups.py model=gradient_boosting model.n_estimators=200

# v5: instantiate() Magic
uv run python src/v5_hydra_instantiate.py model=random_forest preprocessing=mean
uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.learning_rate=0.05

# v6: Advanced Features
uv run python src/v6_hydra_advanced.py model=random_forest
uv run python src/v6_hydra_advanced.py +experiment=quick_test
```

### Dynamic Parameter Addition (v5+)

```bash
# Add new parameter not in YAML (+ prefix)
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Override existing parameter (no + needed)
uv run python src/v5_hydra_instantiate.py model=random_forest model.n_estimators=500

# Combine: swap component + override + add new
uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.n_estimators=200 +model.subsample=0.8
```

### Multirun Sweeps (v6)

```bash
# Sweep over models
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting

# Sweep over preprocessing methods
uv run python src/v6_hydra_advanced.py --multirun preprocessing=ordinal,onehot,mean

# Grid sweep (3 models x 3 encodings = 9 runs)
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting preprocessing=ordinal,onehot,mean
```

### Experiment Presets (v6)

```bash
# Quick test configuration
uv run python src/v6_hydra_advanced.py +experiment=quick_test

# Full comparison configuration
uv run python src/v6_hydra_advanced.py +experiment=full_comparison

# Override within experiment
uv run python src/v6_hydra_advanced.py +experiment=quick_test model.max_iter=200
```

---

## Project Structure

```
hydra-demo/
├── conf/                                # Configuration files
│   ├── config.yaml                      # Main config (v4/v5/v6)
│   ├── config_v3.yaml                   # Flat config (v3 only)
│   ├── dataset/
│   │   └── titanic.yaml                 # Dataset configuration
│   ├── model/                           # Model config groups
│   │   ├── logistic_regression.yaml
│   │   ├── random_forest.yaml
│   │   └── gradient_boosting.yaml
│   ├── preprocessing/                   # Preprocessing config groups
│   │   ├── ordinal.yaml
│   │   ├── onehot.yaml
│   │   └── mean.yaml
│   └── experiment/                      # Experiment presets
│       ├── quick_test.yaml
│       └── full_comparison.yaml
│
├── src/                                 # Source code
│   ├── utils.py                         # Shared utilities
│   ├── v1_pure_python.py                # Step 1: Hardcoded
│   ├── v2_click.py                      # Step 2: Click CLI
│   ├── v3_hydra_basic.py                # Step 3: Basic Hydra
│   ├── v4_hydra_groups.py               # Step 4: Config groups
│   ├── v5_hydra_instantiate.py          # Step 5: instantiate()
│   └── v6_hydra_advanced.py             # Step 6: Advanced features
│
├── outputs/                             # Generated experiment outputs
├── multirun/                            # Generated multirun outputs
├── pyproject.toml                       # Project metadata
└── README.md                            # This file
```

---

## Key Concepts Summary

### 1. Config Groups
- **Swappable configurations**: `model=random_forest` loads `conf/model/random_forest.yaml`
- **No parameter leaking**: Each YAML has only its own parameters
- **DRY principle**: `random_state: ${random_state}` references top-level config

### 2. `_target_` and `instantiate()`
- **`_target_`**: Specifies the fully-qualified class name
- **`instantiate()`**: Dynamically imports class and instantiates with config as kwargs
- **Zero boilerplate**: No imports, registries, or if/elif chains needed

### 3. Dynamic Parameters (`+` syntax)
- **Add new params**: `+model.warm_start=true` adds param not in YAML
- **Override existing**: `model.n_estimators=500` overrides YAML value
- **No code changes**: Works immediately without modifying source

### 4. Experiment Tracking
- **Auto-saved configs**: `.hydra/config.yaml`, `.hydra/overrides.yaml`
- **Custom artifacts**: Metrics, reports, encoder mappings, feature importance
- **Reproducibility**: Every run is fully documented and reproducible

### 5. Defaults List
```yaml
defaults:
  - dataset: titanic              # Loads conf/dataset/titanic.yaml
  - preprocessing: ordinal         # Loads conf/preprocessing/ordinal.yaml
  - model: logistic_regression     # Loads conf/model/logistic_regression.yaml
  - _self_                         # Include this file's keys
```

### 6. Config Interpolation
```yaml
# In config.yaml
random_state: 42

# In conf/model/random_forest.yaml
random_state: ${random_state}     # References top-level key
```

---

## Why Hydra?

### Before Hydra (Click)
```python
# 50+ lines of decorators, if/elif chains, manual dispatch
@click.option("--model", type=click.Choice([...]))
@click.option("--max-iter", ...)
@click.option("--n-estimators", ...)
# ... 10 more options ...

def main(model, max_iter, n_estimators, ...):
    if model == "logistic_regression":
        classifier = LogisticRegression(max_iter=max_iter, ...)
    elif model == "random_forest":
        classifier = RandomForest(n_estimators=n_estimators, ...)
    # ... more if/elif ...
```

### After Hydra (v5)
```python
# 3 lines: Load config, instantiate, train
@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    classifier = hydra.utils.instantiate(cfg.model)
    # Done!
```

### The Value Proposition

| Feature | Click/Argparse | Hydra |
|---------|----------------|-------|
| Add new model | Edit code, redeploy | Create YAML file |
| Add new parameter | Edit decorators, redeploy | `+model.param=value` |
| Swap configurations | Manual arg coordination | `model=random_forest` |
| Config saving | Manual implementation | Automatic |
| Experiment tracking | Manual implementation | Automatic |
| Reproducibility | Hope you saved CLI args | `.hydra/overrides.yaml` |
| Code complexity | 100+ lines w/ if/elif | ~15 lines |

---

## Next Steps

1. **Run the demos**: Start with v1 and progress through v6
2. **Inspect outputs**: Check `outputs/` to see auto-saved configs and artifacts
3. **Try dynamic params**: Add `+model.warm_start=true` to any v5/v6 command
4. **Create your own model**: Add a YAML file to `conf/model/` and run it
5. **Multirun sweeps**: Use `--multirun` to compare multiple configs
6. **Read the docs**: [Hydra Documentation](https://hydra.cc/)

---

## Common Pitfalls

### 1. `_convert_: all` Required for Lists

Feature-engine expects Python lists, but Hydra/OmegaConf uses `ListConfig`.

**Wrong**:
```yaml
# conf/preprocessing/ordinal.yaml
variables:
  - sex
  - embarked
```

**Right**:
```yaml
# conf/preprocessing/ordinal.yaml
_convert_: all                    # Converts OmegaConf types to Python types
variables:
  - sex
  - embarked
```

### 2. Adding vs Overriding Parameters

- **Override existing param** (no prefix): `model.n_estimators=500`
- **Add new param** (+ prefix): `+model.warm_start=true`

If you forget `+` when adding a new param, Hydra will error.

### 3. Config Composition Order

The `_self_` entry in defaults list matters:

```yaml
defaults:
  - model: logistic_regression
  - _self_                        # This file's keys override groups above

random_state: 42                  # This overrides model's random_state
```

---

## Verification Commands

Run these to verify everything works:

```bash
# Test each version
uv run python src/v1_pure_python.py
uv run python src/v2_click.py --model random_forest --encoding onehot
uv run python src/v3_hydra_basic.py model.name=random_forest
uv run python src/v4_hydra_groups.py model=random_forest preprocessing=mean
uv run python src/v5_hydra_instantiate.py model=random_forest preprocessing=mean
uv run python src/v6_hydra_advanced.py model=random_forest

# Test dynamic param addition (the "wow" moment)
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Test artifact saving
uv run python src/v5_hydra_instantiate.py model=random_forest
ls outputs/$(ls -t outputs | head -1)/  # Should show metrics.json, encoder_mapping.json, etc.

# Test multirun
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting
```

Expected output:
- Each version should print accuracy and F1 score
- v3+ should create `outputs/` directory with timestamped subdirectories
- v5/v6 should save custom artifacts (metrics, encoder mappings, feature importance)
- v6 multirun should create separate directories for each config combination

---

## Resources

- **Hydra Docs**: https://hydra.cc/
- **instantiate() Guide**: https://hydra.cc/docs/advanced/instantiate_objects/overview/
- **Config Groups**: https://hydra.cc/docs/tutorials/structured_config/config_groups/
- **Multirun**: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

---

## License

MIT

---

## Teaching Notes

This demo is designed to progressively reveal Hydra's value:

1. **v1-v2**: Establish the pain (hardcoding, if/elif, parameter management)
2. **v3**: Introduce structure (YAML, dot notation, auto-save)
3. **v4**: Show composition (config groups, swappable components)
4. **v5**: **The "aha!" moment** (`instantiate()` eliminates everything)
5. **v6**: Production features (multirun, experiments, Compose API)

The key teaching moment is **v5**, where students see:
- Training script shrinks to ~15 lines
- Adding a model = creating one YAML file
- Dynamic parameters via `+` with zero code changes
- Automatic experiment tracking

This progression mirrors real-world adoption: start simple, add complexity only when needed.
