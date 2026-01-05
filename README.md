# constraint_bootstrap_v1

Constraint-first communication bootstrap simulator.

## Project Recap / Contract

- [Final Project Recap (v1.1)](docs/PROJECT_RECAP_v1_1.md)
This recap defines the locked semantics/invariants for lanes, truth/eligibility, QUESTION cost, and De guardrails.

## Run Examples

```powershell
# Basic runs
python -m constraint_bootstrap.demo_bootstrap_v1 --partner prime --steps 60
python -m constraint_bootstrap.demo_bootstrap_v1 --partner mixed --steps 1000

# Experiment with competition and decay
python -m constraint_bootstrap.demo_bootstrap_v1 --partner adversarial --steps 1000 --compete-topk 1 --inhibit-mult 0.9 --decay-rate 0.003
```

## Summary and Comparison

```powershell
# Summarize a single run
python -m constraint_bootstrap.run_summary_v1 --in out.txt

# Compare two runs (learn-on vs frozen)
python -m constraint_bootstrap.run_compare_v1 --learn-on on.txt --frozen off.txt --out-csv compare.csv
```

## Advanced Metrics and Plotting

The `run_metrics_v1` tool computes "Response Efficiency" (accuracy on non-empty events), "Speak-rate", and "Utility" (coverage-adjusted score).

```powershell
# Metrics from a log file
python -m constraint_bootstrap.run_metrics_v1 --in out.txt --out-json metrics.json

# Metrics from a comparison CSV (with plots)
python -m constraint_bootstrap.run_metrics_v1 --csv compare.csv --plot --roll 25
```

## Aggressive Training

The `train_aggressive` command in `qd_shell_v1` allows for rapid learning with boundary drills and a silence penalty to encourage risk-taking.

```powershell
# Run aggressive training with silence penalty
python -m q_ternary.qd_shell_v1 train_aggressive --rounds 10 --batch 100 --silence-penalty 0.1
```
