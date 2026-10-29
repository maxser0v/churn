"""Validate churn data before training."""
import sys

import pandas as pd

COLUMNS = ["tenure_months", "monthly_charges", "contract_type", "support_calls_90d",
           "late_payments_12m", "has_security_addon", "age", "churned"]
MIN_ROWS = 1000
CHURN_RATE_RANGE = (0.05, 0.40)
MAX_DUPLICATE_SHARE = 0.05


def run_checks(df):
    results = []

    def check(name, passed, detail=""):
        results.append((name, bool(passed), detail))

    check("columns match the contract", list(df.columns) == COLUMNS)
    if list(df.columns) != COLUMNS:
        return results
    check("enough rows", len(df) >= MIN_ROWS, f"{len(df)} rows")
    check("no missing values", not df.isna().any().any())
    d = df.apply(pd.to_numeric, errors="coerce")
    check("numeric and finite", d.notna().all().all() and not d.isin([float("inf"), float("-inf")]).any().any())
    check("tenure 0 to 120", d.tenure_months.between(0, 120).all())
    check("monthly charges positive", (d.monthly_charges > 0).all())
    check("contract types allowed", d.contract_type.isin([0, 1, 2]).all())
    for column in ("support_calls_90d", "late_payments_12m"):
        check(f"{column} nonnegative integer", ((d[column] >= 0) & (d[column] % 1 == 0)).all())
    check("security addon binary", d.has_security_addon.isin([0, 1]).all())
    check("age 18 to 110", d.age.between(18, 110).all())
    check("label binary", d.churned.isin([0, 1]).all())
    rate = d.churned.mean()
    check("churn rate plausible", CHURN_RATE_RANGE[0] <= rate <= CHURN_RATE_RANGE[1], f"{rate:.3f}")
    check("duplicate share acceptable", df.duplicated().mean() <= MAX_DUPLICATE_SHARE)
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    df = pd.read_csv(argv[0] if argv else "data/churn.csv")
    results = run_checks(df)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not all(ok for _, ok, _ in results):
        print("Data validation FAILED: the model was NOT trained")
        return 1
    print(f"all {len(results)} data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
