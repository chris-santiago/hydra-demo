# Hydra Teaching Demo: Complete Narrative Script

**Duration:** ~30-45 minutes
**Audience:** ML engineers, data scientists, Python developers
**Goal:** Show the progressive value of Hydra for ML experiment management

---

## Introduction (3 minutes)

**[Presenter Introduction]**

Good [morning/afternoon/evening]! Today I'm going to show you something that fundamentally changed how I approach ML experiments. And I promise you, there's going to be a moment—usually around the 20-minute mark—where this is all going to click, and you'll think, "Wait, THAT'S what Hydra does? Why wasn't I using this before?"

**[Set the stage with a story]**

Picture this: It's 2am. You've been running experiments all day. Your terminal history is 500 lines long. You finally get great results—0.92 F1 score! Your best yet. You celebrate, go to bed, and come back the next morning ready to build on that success.

There's just one problem: You have absolutely no idea what parameters you used.

Was it 200 estimators or 500? Was max_depth 10 or None? Did you use ordinal encoding or one-hot? You scroll through your terminal history. You check your git log. Nothing. Those results are gone. Lost to the void of undocumented experiments.

**[Pause for effect]**

This is the reality for most ML practitioners. We've solved the hard problems—model architectures, training loops, evaluation metrics—but we're still managing experiments like it's 2010. Manually editing code. Copy-pasting entire scripts. Naming files `train_final_v3_actually_final.py`.

Today, we're going to fix that.

**The Journey:**

We're going to train a simple Titanic survival classifier—nothing fancy, just scikit-learn. Same dataset. Same task. But we're going to implement it **six different ways**. Each version will solve the pain points of the previous one. And somewhere around version 5, you're going to see the "aha moment."

By the end of this demo, you'll understand:
- Why configuration management isn't just "nice to have"—it's essential
- How Hydra eliminates the boilerplate that's been cluttering your code
- The real magic of `instantiate()` (and why it changes everything)
- How to add new parameters **without touching your code**
- Why your experiments should be reproducible by default, not by heroic effort

**[Set expectations]**

I know some of you are thinking: "Another config framework? Really?"

Stick with me. This isn't just about YAML files. This is about fundamentally rethinking how we build ML experiments. And I promise, by version 5, you'll see why Hydra has become the de facto standard at places like Meta, FAIR, and across research labs worldwide.

Ready? Let's start at the beginning—the painful beginning.

---

## Part 1: The Pain (7 minutes)

### Version 1: Pure Python — Everything Hardcoded

**[Set the scene]**

This is where every ML project starts. You're prototyping in a notebook, things are working, so you move it to a Python script. Clean. Simple. It works.

**[Run the demo]**

```bash
uv run python src/v1_pure_python.py
```

**[While it runs, show the code]**

Let me show you what this looks like inside:

```python
# Everything is hardcoded
pipeline = Pipeline([
    ("num_imputer", MeanMedianImputer(imputation_method="mean", variables=["age", "fare"])),
    ("cat_imputer", CategoricalImputer(imputation_method="frequent", variables=["sex", "embarked"])),
    ("encoder", OrdinalEncoder(variables=["sex", "embarked"], ignore_format=True)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
```

**[Point to the screen]**

See those parameters? `max_iter=1000`, `random_state=42`, `imputation_method="mean"`. They're all hardcoded. Baked right into the source.

**[Results appear - point them out]**

Great! We got 79% accuracy. Not bad for a quick baseline.

But now...

**[Pause dramatically]**

Your manager walks over: "Hey, can you try this with RandomForest instead? I read a paper that said tree-based models work better for this kind of data."

**[Show the audience the problem]**

What do you do? You open `train.py`. You find this line:

```python
("classifier", LogisticRegression(max_iter=1000, random_state=42)),
```

And you change it to:

```python
("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
```

You run it again. You get 78% accuracy. Slightly worse. You go back. You change it back to LogisticRegression. But wait—what was the max_iter? Was it 1000 or 500? You check your git diff. You realize you've been committing directly to main because it's "just experiments."

**[Let that sink in]**

This is where the pain begins:

❌ **Want to try RandomForest?** Edit the source code.
❌ **Want different parameters?** Edit the source code.
❌ **Want to try 5 different encoders?** Edit the source code 5 times.
❌ **Run 10 experiments?** Good luck remembering what you changed each time.
❌ **Reproduce last week's results?** Better hope you took good notes.

No configuration management. No experiment tracking. No versioning. Just manual edits, scattered notes, and hope.

**[Acknowledge the elephant in the room]**

Now, some of you are probably thinking: "But I write my parameters in comments!" or "I log them to WandB!" Sure. But you're fighting the system. You're adding discipline where the code should enforce it by default.

Let's try to fix this the "obvious" way.

---

### Version 2: Click CLI — The "Obvious" First Step

**[Transition]**

Okay, so hardcoding is clearly a problem. What's the standard solution? CLI arguments! Everyone knows Click or argparse. Let's add some flags.

**[Run the demo with enthusiasm]**

```bash
uv run python src/v2_click.py --model random_forest --encoding onehot
```

**[Point to the command as it runs]**

Look! Now we can change the model from the command line. Much better! Now when your manager asks you to try RandomForest, you just change `--model logistic_regression` to `--model random_forest`. No code edits!

**[Show the results]**

78% accuracy with RandomForest, 79% with logistic regression. Now you can easily try both. Progress!

**[Transition to showing the cost]**

But let's look at what we had to build to get here...

**[Pull up the code - scroll slowly through the decorators]**

```python
@click.command()
@click.option("--model", type=click.Choice(["logistic_regression", "random_forest", "gradient_boosting"]))
@click.option("--encoding", type=click.Choice(["ordinal", "onehot", "mean"]))
@click.option("--test-size", type=float, default=0.2)
@click.option("--random-state", type=int, default=42)
@click.option("--max-iter", type=int, default=1000, help="Max iterations (LogisticRegression)")
@click.option("--n-estimators", type=int, default=100, help="Number of estimators (RF/GBM)")
@click.option("--max-depth", type=int, default=None, help="Max tree depth (RF/GBM)")
@click.option("--learning-rate", type=float, default=0.1, help="Learning rate (GBM)")
def main(model, encoding, test_size, random_state, max_iter, n_estimators, max_depth, learning_rate):
```

**[Stop scrolling, let them absorb it]**

Eight decorators. Eight parameters in the function signature. And we only have three models!

**[Scroll down to show the if/elif chains]**

And then, inside the function:

```python
# Encoder selection
if encoding == "ordinal":
    encoder = OrdinalEncoder(...)
elif encoding == "onehot":
    encoder = OneHotEncoder(...)
elif encoding == "mean":
    encoder = MeanEncoder(...)

# Model selection
if model == "logistic_regression":
    classifier = LogisticRegression(max_iter=max_iter, ...)
elif model == "random_forest":
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, ...)
elif model == "gradient_boosting":
    classifier = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, ...)
```

**[Explain the problems]**

We've traded one problem for several new ones:

❌ **if/elif chains** — Two of them! One for encoders, one for models. Every new model adds 3-4 lines.
❌ **Parameter pollution** — `--max-iter` shows up even when you're using RandomForest (which doesn't have max_iter!)
❌ **No config saving** — What parameters did you use last week? Better hope your shell history goes back that far.
❌ **Coupled code** — The CLI definition, the function signature, and the if/elif logic all have to be updated together.

**[Now the killer example]**

**The "Add a Parameter" Problem:**

**[Slow down, make this personal]**

You're reading the RandomForest docs, and you discover `warm_start=True`. "Interesting," you think. "Let me try that." In an ideal world, you'd just run:

```bash
python train.py --model random_forest --warm-start true
```

But you can't. Because `warm_start` doesn't exist in your CLI yet. So here's what you **actually** have to do:

**[Show each step on screen, one at a time, with increasing frustration]**

```python
# Step 1: Scroll to top of file, add a decorator
@click.option("--warm-start", type=bool, default=False, help="Enable warm start")

# Step 2: Add it to the function signature (now 9 parameters!)
def main(model, encoding, test_size, random_state, max_iter,
         n_estimators, max_depth, learning_rate, warm_start):
         # ↑ Don't forget to add it here too!

# Step 3: Scroll down 50 lines, thread it through the if/elif
if model == "random_forest":
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        warm_start=warm_start,  # ← Finally, add the actual parameter
        random_state=random_state
    )

# Step 4: Save. Commit. Redeploy. Re-run experiments.
```

**[Pause, let them feel the pain]**

Four steps. Three different locations in the code. All to test one parameter.

And the worst part? That parameter only makes sense for RandomForest. But now `--warm-start` shows up in `--help` for **every model**. Users running LogisticRegression see it and wonder "What's that for?" Your CLI is cluttered with irrelevant options.

**[Drive it home - speak directly to the audience]**

How many of you have codebases like this? **[pause for hands]**

How many of you have added "just one more parameter" and watched your main() function signature grow to 15 arguments? **[pause]**

This pattern doesn't scale. And deep down, we all know it.

**[Transition with confidence]**

There has to be a better way. And there is. It's called Hydra.

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

## Part 3: The "Aha!" Moment (12 minutes)

### Version 5: `instantiate()` — The Magic ✨

**[Set up the reveal]**

Okay. We've seen v3 and v4. They're improvements. Config groups are nice. But we still have registries. We still have imports. We still have boilerplate.

Version 5 is different.

**[Dramatic pause - let silence hang for 2 seconds]**

This is the moment that changes everything.

**[Build anticipation]**

I'm going to show you what the training script looks like. And I want you to watch for what's **missing**.

**[Run the demo first]**

```bash
uv run python src/v5_hydra_instantiate.py model=random_forest
```

**[While it's running, show the code slowly]**

Here's the entire training script:

```python
@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Load data
    X_train, X_test, y_train, y_test = load_titanic(
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_state,
        stratify=cfg.dataset.stratify
    )

    # Instantiate everything from config
    encoder = hydra.utils.instantiate(cfg.preprocessing)
    classifier = hydra.utils.instantiate(cfg.model)
    num_imputer = hydra.utils.instantiate(cfg.num_imputer)
    cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)

    # Build and train
    pipeline = Pipeline([
        ("num_imputer", num_imputer),
        ("cat_imputer", cat_imputer),
        ("encoder", encoder),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate and save
    metrics = evaluate_model(pipeline, X_test, y_test)
    log.info(f"Accuracy: {metrics['accuracy']:.4f}")
    save_artifacts(output_dir, pipeline, metrics, X_test, y_test)
```

**[Let that sink in - speak slowly]**

That's it.

The. Entire. Training. Script.

**[Point to the screen]**

Look at what's **NOT** there:
- ❌ No model imports (`from sklearn.ensemble import...`)
- ❌ No registries (`MODEL_REGISTRY = {...}`)
- ❌ No if/elif chains
- ❌ No custom build functions
- ❌ No dispatching logic

Just four lines:
```python
encoder = hydra.utils.instantiate(cfg.preprocessing)
classifier = hydra.utils.instantiate(cfg.model)
num_imputer = hydra.utils.instantiate(cfg.num_imputer)
cat_imputer = hydra.utils.instantiate(cfg.cat_imputer)
```

**[Results come back - point them out]**

There's our 78% accuracy. Same as before. But look at the log file...

**[Show the log file]**

```bash
cat outputs/sklearn.ensemble.RandomForestClassifier/2026-02-10_23-03-37/v5_hydra_instantiate.log
```

Everything logged. Config saved. Artifacts organized. **By default.**

**[Now explain the magic]**

What is `hydra.utils.instantiate()` doing?

**[Walk through it step by step]**

Let's trace what happens when you call:
```python
classifier = hydra.utils.instantiate(cfg.model)
```

**[Show the config]**

The config (`conf/model/random_forest.yaml`) looks like this:
```yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
max_depth: null
random_state: ${random_state}
```

**[Explain each step with hand gestures]**

When `instantiate()` runs, it:

1. **Reads `_target_`**: "Okay, I need to create a `sklearn.ensemble.RandomForestClassifier`"

2. **Imports the class dynamically**:
   ```python
   # Hydra does this for you:
   from sklearn.ensemble import RandomForestClassifier
   ```
   You never write this import. Hydra handles it.

3. **Collects all other keys as kwargs**:
   ```python
   kwargs = {
       "n_estimators": 100,
       "max_depth": None,
       "random_state": 42  # resolved from ${random_state}
   }
   ```

4. **Instantiates and returns**:
   ```python
   return RandomForestClassifier(**kwargs)
   # Same as: RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
   ```

**[Pause for emphasis]**

It's that simple. `_target_` tells it what to build. The rest are arguments. **Config becomes code.**

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

**[Now make it concrete]**

Okay, that was the magic. But let me show you why this matters in practice.

**[Set up the scenario]**

It's Monday morning. You read a paper over the weekend that says SVMs work really well for this kind of classification problem. You want to try it.

**[Click version - speak with frustration]**

In the Click world (v2), you'd have to:
1. Add sklearn.svm.SVC to your imports
2. Add `svm` to your `--model` choices
3. Add SVM-specific parameters (`--C`, `--kernel`)
4. Add another elif branch
5. Commit, redeploy, run

**[Hydra version - speak with satisfaction]**

In Hydra? Watch:

**[Open a new file on screen]**

I create a new file: `conf/model/svm.yaml`

```yaml
_target_: sklearn.svm.SVC
C: 1.0
kernel: rbf
random_state: ${random_state}
```

**[Type slowly, show it's just 4 lines]**

Four lines. That's it.

**[Run it - speak excitedly]**

```bash
uv run python src/v5_hydra_instantiate.py model=svm
```

**[While it runs]**

No Python code changes. No imports. No if/elif. I created a configuration file, and Hydra handles the rest.

**[Results appear]**

There we go! 79% accuracy. SVM works just as well as LogisticRegression for this dataset.

**[Drive the point home]**

What just happened?
- ✅ Added a new model in **30 seconds**
- ✅ Zero code changes
- ✅ Zero imports
- ✅ Zero if/elif branches
- ✅ Config saved automatically
- ✅ Results logged

**[Pause for effect]**

Config IS the code. That YAML file is as powerful as 20 lines of Python would have been.

---

### Dynamic Parameter Addition — The Game Changer

**[Call back to the pain]**

Remember way back in v2, when I showed you the "Add a Parameter" problem? Adding `warm_start=True` to RandomForest required:
- Step 1: Add decorator
- Step 2: Update function signature
- Step 3: Thread through if/elif
- Step 4: Redeploy

Four steps. Multiple file locations. Just to test one parameter.

**[Build suspense]**

Well, with Hydra's `instantiate()`, that problem... doesn't exist.

Watch:

**[Type it out character by character so they see it]**

```bash
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
```

**[Point to the + sign]**

See that `+` before `model.warm_start`? That's the key. It tells Hydra: "Add this parameter, even though it's not in the YAML file."

**[Run it]**

```bash
uv run python src/v5_hydra_instantiate.py model=random_forest +model.warm_start=true
```

**[While it runs, explain]**

What's happening under the hood:
1. Hydra loads `conf/model/random_forest.yaml` (which has `n_estimators`, `max_depth`, etc.)
2. Hydra sees your `+model.warm_start=true` override
3. Hydra **adds** `warm_start: true` to the config
4. `instantiate()` passes everything as kwargs:
   ```python
   RandomForestClassifier(
       n_estimators=100,
       max_depth=None,
       random_state=42,
       warm_start=True  # ← Added dynamically!
   )
   ```

**[Results come back]**

There! 78% accuracy, same as before. But now we know `warm_start` doesn't help for this dataset.

**[Emphasize the contrast]**

Let me be crystal clear about what just happened:

**Click v2 approach:**
- ✅ Edit 3 different parts of code
- ✅ Commit and redeploy
- ✅ 5 minutes of work
- ❌ Parameter shows up for ALL models

**Hydra v5 approach:**
- ✅ Type `+model.warm_start=true` in terminal
- ✅ Run immediately
- ✅ 10 seconds of work
- ✅ Parameter only applies to this run

**[Show more powerful examples]**

This works for any sklearn parameter:

```bash
# Try L1 regularization with LogisticRegression (requires saga solver)
uv run python src/v5_hydra_instantiate.py model=logistic_regression +model.penalty=l1 +model.solver=saga

# Try subsample for GradientBoosting (not in our YAML by default)
uv run python src/v5_hydra_instantiate.py model=gradient_boosting +model.subsample=0.8

# Combine everything: swap model + override existing + add new
uv run python src/v5_hydra_instantiate.py model=gradient_boosting model.n_estimators=200 +model.subsample=0.8 +model.min_samples_split=10
```

**[Let that sink in]**

No code changes. No redeployment. Just explore the parameter space from your terminal.

**[Make it personal]**

How many hours have you spent adding CLI flags that you used once? How many parameters have you copy-pasted because editing the code felt like too much friction?

With Hydra, that friction is gone. Experimentation becomes **effortless**.

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

## Conclusion & Q&A (3 minutes)

**[Bring it home - speak from the heart]**

Alright. Let's bring this full circle.

**[Recap the journey with energy]**

We started at 2am, scrolling through terminal history, trying to remember what parameters gave us 0.92 F1.

We walked through six versions:
1. **v1:** Hardcoded Python - edit code for everything
2. **v2:** Click CLI - better, but if/elif chains and parameter pollution
3. **v3:** Basic Hydra - YAML configs and auto-saving
4. **v4:** Config groups - swappable components, but still have registries
5. **v5:** **The "aha!" moment** ✨ - `instantiate()` eliminates ALL boilerplate
6. **v6:** Production ready - multirun, presets, logging integration

**[Show the comparison table - let them see the difference]**

Here's what changed:

| What You Want To Do | Click/Argparse | Hydra |
|---------------------|----------------|-------|
| Add a new model | Edit code, redeploy (5 min) | Create YAML file (30 sec) |
| Try a new parameter | Edit decorators, redeploy | `+model.param=value` (immediate) |
| Swap configurations | Coordinate 10 CLI flags | `model=random_forest` (one flag) |
| Track experiments | Manual logging, hope you saved args | Automatic, everything saved |
| Reproduce results | Pray you remember | `cat .hydra/overrides.yaml` and re-run |
| Code complexity | 100+ lines of if/elif | ~10 lines of core logic |

**[Pause to let that sink in]**

**[The core insight - speak slowly and clearly]**

The real "aha!" moment isn't just about less code. It's about **rethinking what code is**.

In the old world, Python files were code and YAML files were "just config."

In the Hydra world, **config IS code**. That YAML file you create? It's as powerful as 20 lines of Python. `instantiate()` made that possible.

**[Make it personal - speak directly to them]**

Think about your current projects. How much time do you spend:
- Adding CLI parameters you'll use once?
- Maintaining if/elif chains?
- Copying scripts because you're afraid to break experiments?
- Trying to remember what you ran last week?

**[Pause]**

What if all of that... just went away?

That's what Hydra gives you. Not a config framework. A fundamentally different way to build ML experiments.

---

**[Practical next steps - be concrete]**

Here's what I want you to do this week:

1. **Clone this repo**: `github.com/chris-santiago/hydra-demo`
2. **Run v1 through v6** - experience the journey yourself
3. **Add your own model** - just create a YAML file, takes 2 minutes
4. **Try the `+` syntax** - add parameters without touching code
5. **Read the docs** - [hydra.cc](https://hydra.cc/) has incredible tutorials

**[The challenge]**

I challenge you: take one of your projects and add Hydra to it. Start small. Just one model. Just one config file. See what it feels like when your experiments become **effortless**.

**[Final thought]**

Someone once told me: "The best config system is the one that stays out of your way."

Hydra doesn't just stay out of your way. It actively helps you move faster. It makes the right things easy and the wrong things hard. It makes reproducibility the default, not a heroic effort.

**[Open the floor - with warmth]**

Alright, I'll stop talking now. Let's discuss. What questions do you have?

**[Be ready for common questions:]**
- "Is this overkill for small projects?" (Maybe, but the overhead is minimal once you know it)
- "What about hyperparameter tuning?" (Perfect use case - multirun with Optuna plugin)
- "Can I use this with PyTorch/TensorFlow?" (Absolutely - works with any Python code)
- "What's the performance overhead?" (Negligible - imports happen once at startup)

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
