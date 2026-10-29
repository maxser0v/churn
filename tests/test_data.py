import pandas as pd

from ci import validate_data


def good_frame():
    return pd.read_csv("data/churn.csv")


def failed(df):
    return {name for name, ok, _ in validate_data.run_checks(df) if not ok}


def test_clean_data_passes():
    assert not failed(good_frame())


def test_negative_tenure_is_caught():
    assert failed(good_frame().assign(tenure_months=-3))


def test_invalid_contract_is_caught():
    assert failed(good_frame().assign(contract_type=5))


def test_invalid_label_is_caught():
    assert failed(good_frame().assign(churned=2))


def test_negative_calls_are_caught():
    assert failed(good_frame().assign(support_calls_90d=-1))


def test_invalid_age_is_caught():
    assert failed(good_frame().assign(age=7))


def test_zero_bill_is_caught():
    assert failed(good_frame().assign(monthly_charges=0))


def test_collapsed_labels_are_caught():
    assert failed(good_frame().assign(churned=0))
    assert failed(good_frame().assign(churned=1))


def test_missing_values_are_caught():
    assert failed(good_frame().assign(monthly_charges=float("nan")))


def test_duplicates_are_caught():
    assert failed(pd.concat([good_frame().head(500)] * 3, ignore_index=True))


def test_missing_columns_are_caught():
    assert failed(good_frame().drop(columns=["age"]))


def test_too_few_rows_are_caught():
    assert failed(good_frame().head(100))


def test_negative_payments_are_caught():
    assert failed(good_frame().assign(late_payments_12m=-1))


def test_invalid_addon_is_caught():
    assert failed(good_frame().assign(has_security_addon=2))


def test_infinite_bill_is_caught():
    assert failed(good_frame().assign(monthly_charges=float("inf")))
