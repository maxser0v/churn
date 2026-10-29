"""Block models that fail measured quality requirements."""
import json
import sys

# Measured in 13e: clean recall .754, shuffled .497; F1 noise half-width .047.
MIN_TEST_RECALL = 0.65
MIN_TEST_PRECISION = 0.25
F1_TOLERANCE = 0.06
MIN_SLICE_RECALL = {"month_to_month": 0.75, "long_contract": 0.15}
MIN_SLICE_POSITIVES = 20


def evaluate(candidate, baseline):
    t = candidate["test"]
    need_f1 = round(baseline["test"]["f1"] - F1_TOLERANCE, 3)
    rules = [
        ("test recall >= floor", t["recall"] >= MIN_TEST_RECALL, t["recall"], f">= {MIN_TEST_RECALL}"),
        ("test precision >= floor", t["precision"] >= MIN_TEST_PRECISION, t["precision"], f">= {MIN_TEST_PRECISION}"),
        ("test F1 not below baseline - tolerance", t["f1"] >= need_f1, t["f1"], f">= {need_f1}"),
    ]
    for name, floor in MIN_SLICE_RECALL.items():
        s = candidate["slices"][name]
        if s["n_positive"] < MIN_SLICE_POSITIVES:
            rules.append((f"{name} slice recall", True, s["recall"], f"skipped, only {s['n_positive']} churners"))
        else:
            rules.append((f"{name} slice recall >= floor", s["recall"] >= floor, s["recall"], f">= {floor}"))
    return rules


def main():
    with open("metrics.json") as f:
        candidate = json.load(f)
    with open("baseline/metrics.json") as f:
        baseline = json.load(f)
    rules = evaluate(candidate, baseline)
    for rule, ok, observed, required in rules:
        print(f"{'PASS' if ok else 'FAIL'}  {rule}  {observed}  {required}")
    if all(ok for _, ok, _, _ in rules):
        print("quality gate PASSED")
        return 0
    print("quality gate FAILED: this model must not be shipped")
    return 1


if __name__ == "__main__":
    sys.exit(main())
