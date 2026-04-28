"""
Statistical significance analysis for LLM eval JSON files.

Based on the framework from:
  Miller (2024) "Adding error bars to evals" [arXiv:2411.00640]
  Wolfe (2026) "Applying Statistics to LLM Evaluations"
  https://cameronrwolfe.substack.com/p/stats-llm-evals

Expected JSON format:
  {
    "experiment": "...",
    "model": "...",
    "benchmark": "...",
    "runs": [
      {
        "config": {"name": "...", ...},
        "accuracy": 0.82,
        "correct": 1082,
        "total": 1319,
        "results": [{"index": 1, "gt": "18", "correct": true}, ...]
      },
      ...
    ]
  }

All runs must share the same question indices for paired analysis.
Scores are binary (correct/incorrect) -> Bernoulli SE formula applies.

Analysis performed:
  1. Per-run: mean accuracy, CLT SE, 95% CI
  2. Pairwise paired differences: mean diff, SE, 95% CI, p-value (z-test), correlation
  3. Power analysis: required n to detect each observed pairwise delta
  4. Summary table
"""

import json
import math
import itertools
import argparse
import sys
from pathlib import Path

import numpy as np
from statistics import NormalDist


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Z_95  = 1.96    # z_{0.025}  -> 95% two-sided CI
ALPHA = 0.05
BETA  = 0.20    # power = 1 - beta = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def z_alpha_over_2(alpha: float = ALPHA) -> float:
    return NormalDist().inv_cdf(1 - alpha / 2)

def z_beta_val(beta: float = BETA) -> float:
    return NormalDist().inv_cdf(1 - beta)

def p_value_two_sided(z: float) -> float:
    """Two-sided p-value from a z-statistic."""
    return 2 * NormalDist().cdf(-abs(z))

def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{100 * x:.{decimals}f}%"

def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_pct(lo)}, {fmt_pct(hi)}]"


# ---------------------------------------------------------------------------
# Core statistical functions
# All assume binary (Bernoulli) scores: s_i in {0, 1}
# ---------------------------------------------------------------------------

def clt_se(scores: np.ndarray) -> float:
    """
    CLT standard error for the sample mean.
    For Bernoulli scores this simplifies to sqrt(p*(1-p)/n),
    but we use the general s/sqrt(n) form (identical in the limit).
    Valid when n >= ~200 (see Bowyer et al. 2025 for small-n caveats).
    """
    n = len(scores)
    return float(np.std(scores, ddof=1)) / math.sqrt(n)


def ci_95(mean: float, se: float) -> tuple[float, float]:
    return (mean - Z_95 * se, mean + Z_95 * se)


def run_summary(scores: np.ndarray) -> dict:
    """Per-run statistics."""
    mean = float(np.mean(scores))
    se   = clt_se(scores)
    lo, hi = ci_95(mean, se)
    return {
        "mean": mean,
        "se":   se,
        "ci_lo": lo,
        "ci_hi": hi,
        "n":    len(scores),
    }


def paired_comparison(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Paired difference analysis (recommended when both models see same questions).

    Computes question-level differences d_i = s_a_i - s_b_i, then
    estimates SE of the mean difference using the CLT on those differences.
    This is more powerful than comparing separate CIs because it cancels
    per-question difficulty variance.

    Returns mean diff, SE, 95% CI, z-statistic, two-sided p-value,
    and Pearson correlation between the two score vectors.
    """
    assert len(scores_a) == len(scores_b), "score arrays must be same length for paired analysis"

    diffs    = scores_a - scores_b
    mean_d   = float(np.mean(diffs))
    se_d     = clt_se(diffs)
    lo, hi   = ci_95(mean_d, se_d)

    # z-test: H0: mean_diff = 0
    z_stat   = mean_d / se_d if se_d > 0 else 0.0
    p_val    = p_value_two_sided(z_stat)

    corr     = float(np.corrcoef(scores_a, scores_b)[0, 1])

    return {
        "mean_diff": mean_d,
        "se_diff":   se_d,
        "ci_lo":     lo,
        "ci_hi":     hi,
        "z_stat":    z_stat,
        "p_value":   p_val,
        "significant": p_val < ALPHA,
        "correlation": corr,
    }


def required_n(
    delta: float,
    omega_sq: float,
    sigma_a_sq: float = 0.0,
    sigma_b_sq: float = 0.0,
    K_A: int = 1,
    K_B: int = 1,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """
    Sample size formula from Miller (2024).
    delta     : minimum detectable effect (absolute accuracy difference)
    omega_sq  : Var(x_A - x_B)  — between-question variance of score differences
    sigma_sq  : within-question variance terms (0 for deterministic evals)
    """
    if delta == 0:
        return float("inf")
    za  = z_alpha_over_2(alpha)
    zb  = z_beta_val(beta)
    var = omega_sq + sigma_a_sq / K_A + sigma_b_sq / K_B
    return ((za + zb) ** 2 * var) / (delta ** 2)


def mde_for_n(
    n: int,
    omega_sq: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """Minimum detectable effect for a given n (rearranged sample size formula)."""
    za = z_alpha_over_2(alpha)
    zb = z_beta_val(beta)
    return math.sqrt(((za + zb) ** 2 * omega_sq) / n)


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_scores(run: dict) -> np.ndarray:
    """
    Extract per-question binary scores from a run dict, sorted by question index
    so that paired comparisons align across runs.
    """
    results = sorted(run["results"], key=lambda x: x["index"])
    return np.array([1.0 if r["correct"] else 0.0 for r in results])


def run_label(run: dict) -> str:
    cfg = run["config"]
    name = cfg.get("name", "?")
    extras = {k: v for k, v in cfg.items() if k != "name"}
    if extras:
        extra_str = " | " + ", ".join(f"{k}={v}" for k, v in extras.items())
    else:
        extra_str = ""
    return f"{name}{extra_str}"


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

SEP = "=" * 80

def print_sep(title: str = "") -> None:
    if title:
        print(f"\n{SEP}\n  {title}\n{SEP}")
    else:
        print(SEP)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(path: str) -> None:
    data   = load_json(path)
    runs   = data["runs"]
    labels = [run_label(r) for r in runs]
    scores = [extract_scores(r) for r in runs]
    n      = len(scores[0])

    # Check n for CLT validity
    clt_warning = ""
    if n < 200:
        clt_warning = (
            f"  ⚠  n={n} < 200 — CLT-based CIs may be overconfident.\n"
            "     Consider Bayesian alternatives (Bowyer et al. 2025)."
        )

    print_sep(f"Eval file: {Path(path).name}")
    print(f"  Experiment : {data.get('experiment', '?')}")
    print(f"  Model      : {data.get('model', '?')}")
    print(f"  Benchmark  : {data.get('benchmark', '?')}")
    print(f"  Questions  : {n}")
    print(f"  Runs       : {len(runs)}")
    if clt_warning:
        print(clt_warning)

    # ------------------------------------------------------------------
    # 1. Per-run statistics
    # ------------------------------------------------------------------
    print_sep("1. Per-run accuracy  (CLT SE, 95% CI)")
    summaries = [run_summary(s) for s in scores]
    col_w = max(len(l) for l in labels) + 2
    for label, summary in zip(labels, summaries):
        print(
            f"  {label:<{col_w}}  "
            f"acc = {fmt_pct(summary['mean'])}  "
            f"SE = {fmt_pct(summary['se'])}  "
            f"95% CI = {fmt_ci(summary['ci_lo'], summary['ci_hi'])}  "
            f"n = {summary['n']}"
        )

    # ------------------------------------------------------------------
    # 2. Pairwise paired comparison
    # ------------------------------------------------------------------
    print_sep("2. Pairwise paired comparison  (H₀: mean_diff = 0)")
    pairs = list(itertools.combinations(range(len(runs)), 2))

    all_paired = {}
    for i, j in pairs:
        result = paired_comparison(scores[i], scores[j])
        all_paired[(i, j)] = result

        sig = "✓ SIGNIFICANT" if result["significant"] else "✗ not significant"
        print(
            f"\n  [{labels[i]}]  vs  [{labels[j]}]\n"
            f"    diff      = {fmt_pct(result['mean_diff'])}  "
            f"(A - B, positive = A better)\n"
            f"    SE(diff)  = {fmt_pct(result['se_diff'])}\n"
            f"    95% CI    = {fmt_ci(result['ci_lo'], result['ci_hi'])}\n"
            f"    z-stat    = {result['z_stat']:.3f}\n"
            f"    p-value   = {result['p_value']:.4f}   {sig}  (α = {ALPHA})\n"
            f"    corr(A,B) = {result['correlation']:.3f}"
        )

    # ------------------------------------------------------------------
    # 3. Power analysis
    # ------------------------------------------------------------------
    print_sep("3. Power analysis  (α=0.05, power=0.80)")
    print(
        f"  {'Pair':<{2 * col_w + 10}}  "
        f"{'|delta|':>8}  "
        f"{'n required':>12}  "
        f"{'MDE @ n={n}'.format(n=n):>14}  "
        f"{'detectable?':>12}"
    )
    print("  " + "-" * (2 * col_w + 10 + 50))

    for i, j in pairs:
        result   = all_paired[(i, j)]
        diffs    = scores[i] - scores[j]
        omega_sq = float(np.var(diffs, ddof=1))
        delta    = abs(result["mean_diff"])

        n_req    = required_n(delta, omega_sq) if delta > 0 else float("inf")
        mde      = mde_for_n(n, omega_sq)

        pair_str = f"[{labels[i][:20]}] vs [{labels[j][:20]}]"
        detectable = "yes" if delta >= mde else "no (too small)"

        print(
            f"  {pair_str:<{2 * col_w + 10}}  "
            f"{fmt_pct(delta):>8}  "
            f"{n_req:>12.0f}  "
            f"{fmt_pct(mde):>14}  "
            f"{detectable:>12}"
        )

    # ------------------------------------------------------------------
    # 4. Summary table (machine-friendly)
    # ------------------------------------------------------------------
    print_sep("4. Summary table")
    header = f"  {'Config':<{col_w}}  {'Acc':>8}  {'SE':>8}  {'95% CI':>22}  {'n':>6}"
    print(header)
    print("  " + "-" * (col_w + 50))
    for label, s in zip(labels, summaries):
        print(
            f"  {label:<{col_w}}  "
            f"{fmt_pct(s['mean']):>8}  "
            f"{fmt_pct(s['se']):>8}  "
            f"{fmt_ci(s['ci_lo'], s['ci_hi']):>22}  "
            f"{s['n']:>6}"
        )

    print(f"\n  Significant pairs (p < {ALPHA}):")
    any_sig = False
    for (i, j), result in all_paired.items():
        if result["significant"]:
            any_sig = True
            print(
                f"    {labels[i]}  vs  {labels[j]}  "
                f"→  diff = {fmt_pct(result['mean_diff'])}, "
                f"p = {result['p_value']:.4f}"
            )
    if not any_sig:
        print("    None")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical significance analysis for LLM eval JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more eval JSON files to analyse.",
    )
    args = parser.parse_args()

    for path in args.files:
        try:
            analyze(path)
        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)
            raise