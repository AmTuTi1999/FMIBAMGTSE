# Prompts — all commands to run

All commands are run from the project root (`/home/FMIBAMGTSE` in WSL, or the equivalent on your system).

---

## 0. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: richer SHAP/gradient implementations
pip install shap captum lime
```

---


## 1. Run all experiments (one shot)

```bash
# Full pipeline: Set 1 + Set 2 + tables/figures
python run_all.py

# With explicit config path
python run_all.py --config configs/default.yaml

# Skip individual stages
python run_all.py --skip-set1          # only Set 2 + figures
python run_all.py --skip-set2          # only Set 1 + figures
python run_all.py --skip-figs          # only experiments, no figures
python run_all.py --skip-set1 --skip-set2   # only figures (from existing CSVs)
```

---

## 2. Experiment Set 1 — Marginal / perturbation methods (C1 and C2 violation)

```bash
# All four sub-experiments: E1.1
python experiments/exp1_running_example.py

# With explicit config
python experiments/exp1_running_example.py --config configs/default.yaml
```

**Outputs written to `results/`:**
- `e1_1_running_example.csv` — per-method score on X1 under f1/f2; C2 violation flag

---

## 3. Experiment Set 2 — Gradient methods (C1 violation)

```bash
# All four sub-experiments: E2.1, E2.2, E2.3
python experiments/exp2_gradient.py

# With explicit config
python experiments/exp2_gradient.py --config configs/default.yaml
```

**Outputs written to `results/`:**
- `e2_1_offmanifold.csv` — score(X1|f1) vs score(X1|f3) per gradient method
- `e2_2_gradient_check.csv` — analytic vs numeric ∂f3/∂x1 verification
- `e2_3_manifold_baseline.csv` — IG on-manifold vs off-manifold baseline artifact

---

## 4. Tables and figures

```bash
# Generate Table A, Table B, and both figures
# (reads CSVs from Set 1/2 runs; re-runs Table A inline)
python experiments/results_tables_figures.py

# With explicit config
python experiments/results_tables_figures.py --config configs/default.yaml
```

**Outputs written to `results/`:**
- `table_A.csv` / `table_A.tex` — running-example scores (✓/✗); grouped by method family
- `fig_offmanifold.pdf` — bar chart: gradient score on f1 vs f3 per method

> **Note:** Run Set 1 and Set 2 first so Table B and the figures have CSV data to read. Table A is always computed fresh from the running-example data.

---

## 5. Individual sub-experiments (ad hoc)

You can call the individual `run_*` functions directly from Python:

```python
import sys; sys.path.insert(0, ".")
import numpy as np
from dagfaith.config import load_config, seed_everything, results_dir

cfg = load_config()
seed_everything(cfg["seed"])
rng = np.random.default_rng(cfg["seed"])
out = str(results_dir(cfg))

# E1.1 only
from experiments.exp1_running_example import run_e1_1
run_e1_1(cfg, rng, out)


# E2.1 only
from experiments.exp2_gradient import run_e2_1
run_e2_1(cfg, rng, out)

# E2.2 only
from experiments.exp2_gradient import run_e2_2
run_e2_2(cfg, rng, out)

# E2.3 only
from experiments.exp2_gradient import run_e2_3
run_e2_3(cfg, rng, out)


# Table A only
from experiments.results_tables_figures import build_table_a
build_table_a(cfg, rng, out)


# Off-manifold figure only (requires e2_1 CSV)
from experiments.results_tables_figures import fig_offmanifold
fig_offmanifold(cfg, out)
```


## 6. Configuration overrides

Override any config value by editing `configs/default.yaml` or passing a custom file:

```bash
# Faster run with smaller data (for development)
# Edit configs/default.yaml:
#   running_example.n_samples: 500
#   var_benchmark.configs:
#     - {D: 3, K: 2, n: 200, label: "tiny"}

python run_all.py --config configs/default.yaml
```

Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `seed` | `42` | Global seed; change to test robustness |
| `running_example.n_samples` | `2000` | Larger → lower variance in scores |
| `running_example.delta` | `0.5` | Correlation X1→X2; γ\* = −2βδ |
| `attribution.shap_n_background` | `100` | KernelSHAP background samples |
| `attribution.n_smoothgrad_samples` | `50` | SmoothGrad noise samples |
