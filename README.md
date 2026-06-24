# The Failures of Marginal Influence-Based Attribution Methods for Global Time Series Explanations

Empirical demonstration of the two failure modes the theory predicts, and that an admissible instantiation passes where the others fail.
---

## Failure modes demonstrated

| Condition | Methods | Symptom |
|-----------|---------|---------|
| **(C2) Conditional separation** |Marginal Based Methods, Perturbation Based Methods| Assign nonzero score to a fully-mediated source → spurious edge |
| **(C1) On-manifold evaluation** |Perturbation Based Methods, Gradient Based Methods, Marginal Based Methods| Score depends on `f` off `supp(p)` → different scores for `f1` and `f3` that agree on the manifold |


---

## Repository layout

```
dagfaith/                   Core library
  config.py                 Global seeding, config loader, results_dir
  interface.py              Unified interface: Method.attribute(f, X) → Tensor[B,D,T]

data/
  running_example.py        C1/C2 data generators; f1/f2/f3 (numpy + torch-native)


methods/
  shap_methods.py           KernelSHAP (marginal/conditional), TimeSHAP, ShapTime,
                            CorrelationAttribution (provable C2 violation)
  lime_methods.py           TS-MULE (uniform / matrix / exponential / SAX)
  gradient_methods.py       Saliency, IG, TIG, SmoothGrad, ExpectedAbsGrad

experiments/
  exp1_running_example.py   Set 1 experiments — E1.1
  exp2_gradient.py          Set 2 experiments — E2.1 to E2.4
  results_tables_figures.py Table A/B and figures (PDF export)

configs/
  default.yaml              All hyperparameters (seed, n_samples, tau, …)

results/                    Auto-created; all CSVs, LaTeX tables, PDFs written here
run_all.py                  One-shot runner for every experiment
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core deps: `numpy scipy pandas scikit-learn torch networkx matplotlib tqdm pyyaml`  
Optional (richer method implementations): `shap captum lime`

---

## Quickstart

```bash
# Run everything in one shot
python run_all.py

# Or run experiment sets individually
python experiments/exp1_running_example.py
python experiments/exp2_gradient.py
python experiments/results_tables_figures.py

```

---

## Experiment outputs

All outputs are written to `results/` (created automatically).

| File | Contents |
|------|----------|
| `e1_1_running_example.csv` | Per-method score on X1 under f1/f2; C2 violation flag |
| `e2_1_offmanifold.csv` | Per-method score on X1 under f1 vs f3 (C1 violation) |
| `e2_2_gradient_check.csv` | Analytic vs numeric ∂f3/∂x1 verification |
| `e2_3_manifold_baseline.csv` | IG off-manifold vs on-manifold baseline artifact |
| `table_A.csv` / `.tex` | Running-example scores with ✓/✗; grouped by family |
| `fig_offmanifold.pdf` | Bar chart: gradient score on f1 vs f3 per method |

---

## Configuration

Edit `configs/default.yaml` to change:

- `seed` — global random seed (all results are deterministic at fixed seed)
- `running_example.{beta,delta,n_samples,gamma_sweep}` — running-example parameters
- `attribution.{n_baselines,n_smoothgrad_samples,shap_n_background}` — method tuning

---

## Key design decisions

**Unified interface** — every method implements `Method.attribute(f, X) → Tensor[B, D, T]`. The evaluation harness is method-agnostic; plugging in a new method requires only implementing this one method.

**Autograd-compatible models** — `data/running_example.py` provides both numpy and torch-native versions of f1/f2/f3. Perturbation methods (SHAP, TS-MULE) use the numpy-wrapped version; gradient methods (Saliency, IG, TIG, SmoothGrad) require the torch-native version so autograd propagates through the model.
