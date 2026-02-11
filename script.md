# Hydra Teaching Demo: Complete Narrative Script

**Duration:** ~30-45 minutes
**Audience:** ML engineers, data scientists, Python developers
**Goal:** Show the progressive value of Hydra for ML experiment management

---

## Introduction (2 minutes)

**[Presenter Introduction]**

Today I'm going to show you why Hydra has become essential for ML experiment management. We'll start with pure Python and progressively evolve toward full Hydra, showing you the exact moment when everything clicks.

**The Setup:**

We're going to train a simple Titanic survival classifier using scikit-learn. Same dataset, same task — but we'll implement it six different ways. Each version solves problems from the previous one, until we reach what I call the "aha moment."

By the end, you'll understand:
- Why configuration management matters
- How Hydra eliminates boilerplate
- The magic of `instantiate()`
- Dynamic parameter addition without code changes

Let's dive in.

---

## Part 1: The Pain (5 minutes)

### Version 1: Pure Python — Everything Hardcoded

**[Run the demo]**

```bash
uv run python src/v1_pure_python.py
```

**[Show the code]**

```python
# Everything is hardcoded
pipeline = Pipeline([
    ("num_imputer", MeanMedianImputer(imputation_method="mean", ...)),
    ("cat_imputer", CategoricalImputer(imputation_method="frequent", ...)),
    ("encoder", OrdinalEncoder(...)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
```

**[Explain the pain]**

This is where most ML projects start. Everything works, but:

❌ Want to try RandomForest? **Edit the source code.**
❌ Want different parameters? **Edit the source code.**
❌ Run 10 experiments? **Good luck remembering what you changed each time.**

No configuration management. No experiment tracking. Just manual edits and hope.

---

### Version 2: Click CLI — The "Obvious" First Step

**[Run the demo]**

```bash
uv run python src/v2_click.py --model random_forest --encoding onehot
```

**[Show the code - the decorators]**

```python
@click.command()
@click.option("--model", type=click.Choice([...]))
@click.option("--max-iter", type=int, default=1000)
@click.option("--n-estimators", type=int, default=100)
# ... 10 more options ...
def main(model, max_iter, n_estimators, ...):
```

**[Explain the problem]**

Now we can change models via CLI. Better! But look at what we have to maintain:

❌ **if/elif chains in two places** (encoder selection + model selection)
❌ **All parameters exposed** (--max-iter shows for RandomForest, even though it doesn't use it)
❌ **No config saving** (what parameters did I use last week?)

**The "Add a Parameter" Problem:**

Let's say you want to add `warm_start=True` to RandomForest. Here's what you have to do:

```python
# Step 1: Add decorator
@click.option("--warm-start", type=bool, default=False)

# Step 2: Add to function signature
def main(model, ..., warm_start):

# Step 3: Thread through if/elif
if model == "random_forest":
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        warm_start=warm_start,  # Add here
        ...
    )

# Step 4: Redeploy
```

Four steps. Edit code. Redeploy. This is what Hydra eliminates.

---

## Part 2: Enter Hydra (5 minutes)

### Version 3: Basic Hydra — Structured Configuration

**[Show the YAML config]**

```yaml
# conf/config_v3.yaml
random_state: 42
test_size: 0.2

model:
  name: logistic_regression
  max_iter: 1000
  n_estimators: 100
```

**[Run the demo]**

```bash
uv run python src/v3_hydra_basic.py model.name=random_forest
```

**[Explain the improvements]**

✅ **Structured YAML configuration** (dot notation for overrides)
✅ **Auto-saved in outputs/** (every run documented)
✅ **Cleaner CLI** (no more --flags everywhere)

But... **we still have if/elif chains**:

```python
if cfg.model.name == "logistic_regression":
    classifier = LogisticRegression(...)
elif cfg.model.name == "random_forest":
    classifier = RandomForestClassifier(...)
```

And the config is flat — no composition yet.

---

### Version 4: Config Groups — Swappable Components

**[Show the config structure]**

```
conf/
├── config.yaml              # Main config
├── model/
│   ├── logistic_regression.yaml
│   ├── random_forest.yaml
│   └── gradient_boosting.yaml
└── preprocessing/
    ├── ordinal.yaml
    ├── onehot.yaml
    └── mean.yaml
```

**[Show a model config]**

```yaml
# conf/model/random_forest.yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
max_depth: null
random_state: ${random_state}
```

**[Run the demo]**

```bash
# Swap the entire model config
uv run python src/v4_hydra_groups.py model=random_forest

# Swap both model and preprocessing
uv run python src/v4_hydra_groups.py model=gradient_boosting preprocessing=mean
```

**[Explain config groups]**

This is powerful! `model=random_forest` loads `conf/model/random_forest.yaml` and **replaces the entire model config**.

✅ **No parameter leaking** (random_forest.yaml only has RF params)
✅ **DRY principle** (`random_state: ${random_state}` references top-level config)
✅ **Composition** (mix and match components)

**But we still have problems:**

```python
# Still need manual registries
MODEL_REGISTRY = {
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    ...
}

# Still importing all classes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Still need custom build function
def build_from_config(cfg, registry):
    cls = registry[cfg._target_]
    return cls(**cfg)
```

This is where most tutorials stop. But we can do better.

---

## Part 3: The "Aha!" Moment (10 minutes)

### Version 5: `instantiate()` — The Magic ✨

**[Dramatic pause]**

This is the moment that changes everything. Watch this:

**[Show the training script]**

```python
@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Load data
    X_train, X_test, y_train, y_test = load_titanic(...)

    # THIS IS IT - NO IMPORTS, NO REGISTRIES, NO IF/ELIF!
    encoder = hydra.utils.instantiate(cfg.preprocessing)
    classifier = hydra.utils.instantiate(cfg.model)
    num_imputer = hydra.utils.instantiate(cfg.num_imputer)
    cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

    # Build and train
    pipeline = Pipeline([...])
    pipeline.fit(X_train, y_train)
```

**[Let that sink in]**

That's it. The entire training script. **No model imports. No registries. No if/elif chains.**

**What just happened?**

`hydra.utils.instantiate()` reads the `_target_` key and:

1. Imports the class dynamically (`sklearn.ensemble.RandomForestClassifier`)
2. Passes remaining keys as `**kwargs`
3. Returns the instantiated object

**[Show before/after comparison]**

**Before (v4):**
```python
# 50+ lines of imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from feature_engine.encoding import OrdinalEncoder, OneHotEncoder, MeanEncoder
...

# Manual registry
MODEL_REGISTRY = {...}

# Custom build function
def build_from_config(cfg, registry):
    ...

# Still hardcoded imputers
num_imputer = MeanMedianImputer(imputation_method="mean", ...)
cat_imputer = CategoricalImputer(imputation_method="frequent", ...)
```

**After (v5):**
```python
# Zero imports of model/encoder classes
# Zero registries
# Zero if/elif chains

# Everything from config
encoder = hydra.utils.instantiate(cfg.preprocessing)
classifier = hydra.utils.instantiate(cfg.model)
num_imputer = hydra.utils.instantiate(cfg.num_imputer)
cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)
```

**[Key insight - v4→v5 improvement]**

Notice in v4, imputers were still hardcoded in Python:
```python
# v4: Still hardcoded!
num_imputer = MeanMedianImputer(imputation_method="mean", variables=["age", "fare"])
```

In v5, **everything** is instantiated from config. This is the full power of `instantiate()`.

---

### Adding a New Model — The Power Demo

**[Show the process]**

Want to add an SVM? **Just create a YAML file:**

```yaml
# conf/model/svm.yaml
_target_: sklearn.svm.SVC
C: 1.0
kernel: rbf
random_state: ${random_state}
```

**[Run it]**

```bash
uv run python src/v5_hydra_instantiate.py model=svm
```

**[Emphasize]**

Zero code changes. Just created a config file. **Config IS the code.**

---

### Dynamic Parameter Addition — The Game Changer

**[Set up the problem]**

Remember the Click example? Adding `warm_start` required 4 steps and code edits.

Watch this:

```bash
# Add a parameter NOT in the YAML (+ prefix)
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
```

**[Run it and show it works]**

**[Explain the magic]**

The `+` tells Hydra: "Add this key even though it's not in the YAML."

Since `instantiate()` passes all keys as `**kwargs`, it goes straight to:
```python
RandomForestClassifier(n_estimators=100, warm_start=True, ...)
```

**No code changes. No redeployment. Just run it.**

**[Show more examples]**

```bash
# Add multiple params
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Combine: swap + override + add new
uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.n_estimators=200 +model.subsample=0.8
```

---

### Override Syntax: The Full Trifecta

**[Show the three modes]**

Hydra actually supports three override modes:

```bash
# 1. Override existing (errors if key doesn't exist)
model.n_estimators=500

# 2. Add new (+ prefix, errors if already exists)
+model.warm_start=true

# 3. Force set (++ prefix, works either way)
++model.n_estimators=500
```

| Syntax | Behavior | Use Case |
|--------|----------|----------|
| `key=value` | Override existing | Standard tuning |
| `+key=value` | Add new | Extending config |
| `++key=value` | Force set | Scripts where key may/may not exist |

The `++` is useful for automation where you're not sure if a key exists.

---

### Experiment Tracking & Artifacts

**[Show what Hydra saves]**

Every run creates a timestamped directory:

```bash
outputs/sklearn.ensemble.RandomForestClassifier/2026-02-10_23-03-37/
├── .hydra/
│   ├── config.yaml          # Full resolved config
│   ├── overrides.yaml       # What you typed on CLI
│   └── hydra.yaml           # Hydra metadata
├── v5_hydra_instantiate.log # Auto-captured logs
├── metrics.json             # Your custom artifacts
├── classification_report.txt
├── encoder_mapping.json
└── feature_importance.csv
```

**[Explain the value]**

✅ **Reproducibility:** `cat .hydra/overrides.yaml` shows exactly what you ran
✅ **Organization:** Outputs grouped by model type and timestamp
✅ **Logging:** Python `logging` auto-captured to `.log` files
✅ **Artifacts:** Everything saved in one place

**[Show logging integration]**

```python
import logging
log = logging.getLogger(__name__)

@hydra.main(...)
def main(cfg):
    log.info("Training model...")  # Appears in console AND .log file
```

Hydra auto-configures Python's logging. No more empty log files!

---

## Part 4: Production Features (8 minutes)

### Version 6: Advanced Features

**[Transition]**

v5 showed you the core magic. v6 shows you production features.

---

#### 1. Multirun Sweeps

**[Run the demo]**

```bash
# Sweep over 3 models (runs sequentially)
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting
```

**[Show the output structure]**

```
multirun/2026-02-10_23-03-49/
├── model=logistic_regression/
├── model=random_forest/
└── model=gradient_boosting/
```

Each gets its own directory with full outputs.

**[Important note]**

By default, jobs run **sequentially** (one after another). For parallel execution, you need a launcher plugin like `hydra-joblib-launcher`.

**[Grid sweeps]**

```bash
# 3 models × 3 encodings = 9 runs
uv run python src/v6_hydra_advanced.py --multirun \
  model=logistic_regression,random_forest,gradient_boosting \
  preprocessing=ordinal,onehot,mean
```

---

#### 2. Experiment Presets

**[Show the concept]**

Sometimes you have common configurations you want to reuse:

```yaml
# conf/experiment/quick_test.yaml
# @package _global_
defaults:
  - override /model: logistic_regression
  - override /preprocessing: ordinal

dataset:
  test_size: 0.3

model:
  max_iter: 500
```

**[Run it]**

```bash
uv run python src/v6_hydra_advanced.py +experiment=quick_test
```

One command loads an entire pre-configured setup. Great for:
- Quick debugging runs
- Reproduction setups
- Standard baselines

---

#### 3. Custom Output Directory Structure

**[Show the config]**

```yaml
# In conf/config.yaml
hydra:
  run:
    dir: outputs/${model._target_}/${now:%Y-%m-%d_%H-%M-%S}
```

This organizes outputs by model type:

```
outputs/
├── sklearn.ensemble.RandomForestClassifier/
│   ├── 2026-02-10_14-30-22/
│   └── 2026-02-10_15-45-10/
└── sklearn.linear_model.LogisticRegression/
    └── 2026-02-10_14-35-55/
```

Easy to compare all RandomForest runs vs all LogisticRegression runs.

---

## Part 5: Debugging & Best Practices (5 minutes)

### Debugging Your Config

**[Show the built-in flags]**

Hydra has debugging tools that require **zero code changes**:

```bash
# Print the final composed config (without running)
uv run python src/v5_hydra_instantiate.py --cfg job

# See how defaults were composed
uv run python src/v5_hydra_instantiate.py --info defaults-tree

# Show help with available options
uv run python src/v5_hydra_instantiate.py --help
```

**[Run --cfg job example]**

This is invaluable for understanding config composition.

---

### Common Pitfalls & Solutions

#### 1. Understanding `version_base=None`

```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
```

- `version_base=None` → Latest Hydra behavior (recommended)
- `version_base="1.3"` → Lock to Hydra 1.3 compatibility
- Omitting it → Deprecation warning

**Use `version_base=None` for new projects.**

---

#### 2. `_convert_: all` for Native Python Types

Some libraries expect native Python lists, not OmegaConf's `ListConfig`:

```yaml
# WRONG - will cause issues
_target_: feature_engine.encoding.OrdinalEncoder
variables:
  - sex
  - embarked

# RIGHT - converts to native Python types
_target_: feature_engine.encoding.OrdinalEncoder
_convert_: all
variables:
  - sex
  - embarked
```

**Rule of thumb:** If your library expects lists/dicts, add `_convert_: all`.

---

#### 3. Working Directory Changes

Hydra changes the working directory to the output directory:

```python
# WRONG - will fail!
open("data/file.csv")  # FileNotFoundError

# RIGHT - get original project directory
from hydra.utils import get_original_cwd
import os

data_path = os.path.join(get_original_cwd(), "data", "file.csv")
open(data_path)
```

This demo doesn't hit this because we use `fetch_openml()`.

---

#### 4. Override Syntax Errors

```bash
# Override existing (errors if doesn't exist)
model.n_estimators=500

# Add new (errors if already exists)
+model.warm_start=true
```

Forgetting the `+` when adding new keys is a common mistake.

---

## Part 6: Beyond This Demo (3 minutes)

### Using Hydra in Jupyter Notebooks

The `@hydra.main()` decorator is for scripts. For notebooks, use the **Compose API**:

```python
from hydra import compose, initialize

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config", overrides=["model=random_forest"])

    # Use instantiate as usual
    classifier = hydra.utils.instantiate(cfg.model)
```

**Important:** Compose API **does NOT** create output directories or configure logging. It only composes the config object.

---

### Structured Configs (Type Safety)

This demo uses YAML files, but Hydra also supports **dataclass-based configs**:

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    _target_: str = "sklearn.ensemble.RandomForestClassifier"
    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42

cs = ConfigStore.instance()
cs.store(name="random_forest", node=ModelConfig, group="model")
```

**Benefits:**
- ✅ Type checking at runtime
- ✅ IDE auto-completion
- ✅ Better error messages

**When to use:**
- Large projects with many configs
- Teams prioritizing type safety
- When you want IDE support

See the README "Further Reading" section for more.

---

### Custom OmegaConf Resolvers

You can register custom functions for dynamic values:

```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
```

Then in YAML:
```yaml
hidden_size: ${mul:${n_features},2}
```

Useful for math operations, environment variables, path construction, etc.

---

## Conclusion & Q&A (2 minutes)

**[Recap the journey]**

We started with hardcoded Python, added Click CLI, then progressively adopted Hydra:

1. **v3:** Structured YAML configs
2. **v4:** Composable config groups
3. **v5:** The "aha!" moment with `instantiate()` 🎉
4. **v6:** Production features (multirun, presets)

**The Value Proposition:**

| Feature | Click/Argparse | Hydra |
|---------|----------------|-------|
| Add new model | Edit code, redeploy | Create YAML file |
| Add new parameter | Edit decorators, redeploy | `+model.param=value` |
| Swap configurations | Manual arg coordination | `model=random_forest` |
| Config saving | Manual implementation | Automatic |
| Experiment tracking | Manual implementation | Automatic |
| Code complexity | 100+ lines w/ if/elif | ~10 lines core logic |

**The "aha!" moment:**

`instantiate()` eliminates all boilerplate. Training scripts become config-driven. Adding models is just creating YAML files. **Config IS the code.**

---

**[Call to action]**

Next steps:
1. Clone the repo: `github.com/chris-santiago/hydra-demo`
2. Run through all six versions yourself
3. Try adding your own model (just create a YAML!)
4. Experiment with `+` syntax for dynamic parameters
5. Read the [official Hydra docs](https://hydra.cc/)

**[Open for questions]**

Questions?

---

## Appendix: Quick Reference Commands

```bash
# Run each version
uv run python src/v1_pure_python.py
uv run python src/v2_click.py --model random_forest
uv run python src/v3_hydra_basic.py model.name=random_forest
uv run python src/v4_hydra_groups.py model=random_forest preprocessing=mean
uv run python src/v5_hydra_instantiate.py model=random_forest preprocessing=mean
uv run python src/v6_hydra_advanced.py model=random_forest

# Dynamic parameter addition
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Debugging
uv run python src/v5_hydra_instantiate.py --cfg job
uv run python src/v5_hydra_instantiate.py --info defaults-tree

# Multirun
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting

# Experiment presets
uv run python src/v6_hydra_advanced.py +experiment=quick_test
uv run python src/v6_hydra_advanced.py +experiment=full_comparison
```

---

## Notes for Presenters

**Timing tips:**
- If short on time, combine v1+v2 into "The Pain" (3 min total)
- If very short, skip v3 and go straight to v4
- The critical section is v5 (the "aha!" moment) - don't rush this

**Common questions:**
- "Is this overkill for small projects?" → Maybe, but once you learn it, the overhead is minimal
- "What about CI/CD?" → Hydra works great with automation, especially `++` syntax
- "Performance overhead?" → Negligible - the import happens once at startup

**Demo tips:**
- Have all commands in a separate terminal buffer ready to paste
- Show the log files with `cat outputs/.../script.log` to prove logging works
- If doing live coding, have the SVM example pre-written as a backup

**Engagement points:**
- After v2: "How many of you have this exact Click setup?"
- After v5: "How many lines of if/elif did we just delete?"
- During multirun: "Imagine doing this for 100 hyperparameter combinations"
