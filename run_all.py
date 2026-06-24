"""Run all experiments end-to-end and generate results.

Usage:
    python run_all.py [--config configs/default.yaml]

Order:
    1. Experiment Set 1 (marginal/perturbation methods — C2 violation)
    2. Experiment Set 2 (gradient methods — C1 violation)
    3. Tables and figures (Tasks 7)
"""
import argparse
import sys
import os

def main():
    """Parse CLI flags and run experiment sets 1, 2, and the figures pipeline in order."""
    parser = argparse.ArgumentParser(description="Run all DAGFaith experiments")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--skip-set1", action="store_true")
    parser.add_argument("--skip-set2", action="store_true")
    parser.add_argument("--skip-figs",  action="store_true")
    args = parser.parse_args()

    # Ensure project root is on sys.path
    root = os.path.dirname(__file__)
    sys.path.insert(0, root)

    config_flag = ["--config", args.config]

    if not args.skip_set1:
        print("=" * 60)
        print("EXPERIMENT SET 1 — Marginal methods")
        print("=" * 60)
        from experiments.exp1_running_example import main as run1
        sys.argv = ["exp1"] + config_flag
        run1()

    if not args.skip_set2:
        print("\n" + "=" * 60)
        print("EXPERIMENT SET 2 — Gradient methods")
        print("=" * 60)
        from experiments.exp2_gradient import main as run2
        sys.argv = ["exp2"] + config_flag
        run2()

    if not args.skip_figs:
        print("\n" + "=" * 60)
        print("RESULTS — Tables and figures")
        print("=" * 60)
        from experiments.results_tables_figures import main as run_figs
        sys.argv = ["figs"] + config_flag
        run_figs()

    print("\n" + "=" * 60)
    print("All experiments complete. Check the results/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
