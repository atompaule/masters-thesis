import argparse
import json
import math
import sys
from collections import defaultdict


def load_results(filepath):
    """Loads results from a JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if "results" not in data:
                print(f"Error: 'results' key not found in {filepath}")
                sys.exit(1)
            return data["results"]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def align_results(baseline_results, experiment_results):
    """Aligns results by question text."""
    # Create a map of question -> correct for baseline
    baseline_map = {}
    for res in baseline_results:
        q = res.get("question", "").strip()
        baseline_map[q] = res.get("correct", False)

    # Align experiment results
    aligned_data = []

    missing_in_baseline = 0

    for res in experiment_results:
        q = res.get("question", "").strip()
        if q in baseline_map:
            aligned_data.append(
                {
                    "question": q,
                    "baseline_correct": baseline_map[q],
                    "experiment_correct": res.get("correct", False),
                }
            )
        else:
            missing_in_baseline += 1

    if missing_in_baseline > 0:
        print(
            f"Warning: {missing_in_baseline} questions from experiment file were not found in baseline file."
        )

    return aligned_data


def mcnemar_test(aligned_data):
    """Calculates McNemar's test statistics."""
    # Contingency Table
    #                Experiment Correct  Experiment Wrong
    # Baseline Correct      Yes/Yes (a)      Yes/No (b)
    # Baseline Wrong        No/Yes (c)       No/No (d)

    a = 0  # Both correct
    b = 0  # Baseline correct, Experiment wrong
    c = 0  # Baseline wrong, Experiment correct
    d = 0  # Both wrong

    for item in aligned_data:
        base = item["baseline_correct"]
        exp = item["experiment_correct"]

        if base and exp:
            a += 1
        elif base and not exp:
            b += 1
        elif not base and exp:
            c += 1
        elif not base and not exp:
            d += 1

    total = a + b + c + d
    print(f"\nContingency Table (n={total}):")
    print(f"{'':<20} | {'Exp Correct':<12} | {'Exp Wrong':<12} | {'Total':<10}")
    print("-" * 65)
    print(f"{'Baseline Correct':<20} | {a:<12} | {b:<12} | {a+b:<10}")
    print(f"{'Baseline Wrong':<20} | {c:<12} | {d:<12} | {c+d:<10}")
    print("-" * 65)
    print(f"{'Total':<20} | {a+c:<12} | {b+d:<12} | {total:<10}")

    print(f"\nAccuracy Baseline:   {(a+b)/total:.4f}")
    print(f"Accuracy Experiment: {(a+c)/total:.4f}")

    # McNemar's Test
    # Statistic = (b - c)^2 / (b + c)
    # If b + c < 25, exact binomial test is preferred, but Chi-squared is standard approximation.

    discordant_sum = b + c
    if discordant_sum == 0:
        print(
            "\nMcNemar's Test: Not applicable (no discordant pairs). Models are identical on these samples."
        )
        return

    chi2 = (abs(b - c) - 1) ** 2 / discordant_sum  # With continuity correction

    # Calculate p-value from Chi-squared distribution with 1 degree of freedom
    # We can use a simple approximation or standard library if available, but for now
    # let's try to be self-contained or use scipy if possible.

    try:
        from scipy.stats import chi2 as chi2_dist

        p_value = chi2_dist.sf(chi2, 1)
        print(f"\nMcNemar's Test (with continuity correction):")
        print(f"Chi-squared statistic: {chi2:.4f}")
        print(f"p-value: {p_value:.4e}")

        if p_value < 0.05:
            print("Result: Statistically SIGNIFICANT difference (p < 0.05)")
        else:
            print("Result: NO statistically significant difference (p >= 0.05)")

    except ImportError:
        print("\nMcNemar's Test (Chi-squared statistic):")
        print(f"Chi-squared statistic: {chi2:.4f}")
        print("(Install scipy for automatic p-value calculation)")
        print("Critical value for p=0.05, df=1 is 3.841")
        if chi2 > 3.841:
            print("Result: Statistically SIGNIFICANT difference (chi2 > 3.841)")
        else:
            print("Result: NO statistically significant difference (chi2 <= 3.841)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform McNemar's significance test on two GSM8k eval_results.json files."
    )
    parser.add_argument("baseline", help="Path to the baseline eval_results.json")
    parser.add_argument("experiment", help="Path to the experiment eval_results.json")

    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    base_res = load_results(args.baseline)

    print(f"Loading experiment: {args.experiment}")
    exp_res = load_results(args.experiment)

    aligned = align_results(base_res, exp_res)

    if not aligned:
        print("No overlapping questions found.")
        sys.exit(1)

    mcnemar_test(aligned)
