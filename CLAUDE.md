# CLAUDE.md

> Repo-specific context for Claude Code. Loaded every session.
> Keep this under ~120 lines. Document what Claude gets wrong, not comprehensive manuals.

---

## Project Overview

**Purpose:** Teaching demo that progressively evolves an ML training script from hardcoded Python → Click CLI → Hydra, culminating in `instantiate()` eliminating all boilerplate.
**Stack:** Python 3.12, uv, Hydra 1.3, scikit-learn, feature-engine, Click
**Structure:**
```
src/              # Six versioned scripts (v1–v6), plus shared utils.py
conf/             # Hydra config groups: model/, preprocessing/, dataset/, experiment/
data/             # titanic.csv (local, not fetched from network)
.project-log/     # Session journal (journal.jsonl + helper scripts)
outputs/          # Auto-generated run artifacts (gitignored)
multirun/         # Auto-generated multirun sweep outputs (gitignored)
```

---

## Environment & Commands

```bash
# Setup
uv sync

# Run any version
uv run python src/v1_pure_python.py
uv run python src/v5_hydra_instantiate.py model=random_forest

# Lint
uv run ruff check src/
uv run ruff format src/

# Multirun sweep (v6)
uv run python src/v6_hydra_advanced.py --multirun model=logistic_regression,random_forest,gradient_boosting
```

> No Makefile or CI. All commands use `uv run`.

---

## Code Style

- Match the verbosity and docstring style of the target version file — v1–v3 are heavily commented for teaching; v4–v6 are leaner
- Never add dependencies without asking first; deps are teaching-scoped (scikit-learn, feature-engine, Hydra, Click only)
- YAML keys in `conf/` use `_target_` and `_convert_` — these are Hydra reserved keys, not project conventions; do not rename them
- New model configs go in `conf/model/<name>.yaml`; new preprocessing configs in `conf/preprocessing/<name>.yaml` — never hard-code choices in Python

---

## Architecture Notes

- **Two config roots**: `conf/config_v3.yaml` (flat, for v3 only) and `conf/config.yaml` (defaults list, for v4/v5/v6). Don't mix them.
- **`utils.py` imports**: Scripts import `from utils import ...` without a package prefix — this works because `uv run` sets `src/` as the working directory context via Hydra's `@hydra.main`.
- **`_target_` pattern**: Model and preprocessor YAML files use `_target_` for `hydra.utils.instantiate()`. Adding a new model = new YAML in `conf/model/`, zero code changes.
- **`+` syntax for ad-hoc params**: `+model.warm_start=true` appends kwargs not in YAML. This is intentional and the key teaching point of v5.

---

## Known Gotchas

- Hydra changes the working directory to the output dir at runtime. `utils.py` loads the CSV using `Path(__file__).parent.parent / "data"` to avoid path issues — always use absolute paths relative to `__file__` for data access.
- `conf/config.yaml` has a `hydra.run.dir` override that routes outputs to `outputs/${model._target_}/${timestamp}`. Don't remove this — it's a deliberate teaching feature of v6.
- There are no automated tests. Verification is done by running the scripts and checking printed accuracy/F1 output.
- `environment.yaml` is a conda lockfile artifact — the project uses `uv` + `pyproject.toml` as the authoritative dependency source.

---

## Journal — Proactive Logging

When `.project-log/journal.jsonl` exists, propose logging at natural pauses — not mid-investigation. Always ask first; full draft only after user confirms.

**Auto-propose these types:**

| Pattern | Type | When |
|---------|------|------|
| User confirms a direction | `decision` | After "I agree", "let's do X", "go with that" |
| Unexpected finding | `discovery` | When exploration changes understanding or approach |
| Bug/inconsistency found | `issue` | After identifying and explaining a problem |
| Bug fixed and verified | `resolution` | After fix confirmed working |
| Root cause understood | `lesson` | After explaining *why* something broke — ask "should I log this as a lesson?" |
| Results interpreted | `experiment` | When verdict is clear |

**Do not auto-propose:** `/checkpoint`, `/resume`, `/log-commit`, `/research-note`, `/research-report`, read skills, `hypothesis`, `post_mortem`, `memo`.

**Rules:** One proposal per event. Don't re-propose if declined. Chain issue→resolution→lesson at completion, not as three interruptions.
