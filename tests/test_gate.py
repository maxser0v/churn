import copy

from ci import quality_gate

BASELINE = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438}}
GOOD = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438},
        "slices": {"month_to_month": {"recall": 0.899, "n_positive": 129},
                   "long_contract": {"recall": 0.263, "n_positive": 38}}}


def failed(candidate):
    return [name for name, ok, _, _ in quality_gate.evaluate(candidate, BASELINE) if not ok]


def test_baseline_passes():
    assert not failed(GOOD)


def test_low_recall_is_blocked():
    c = copy.deepcopy(GOOD)
    c["test"]["recall"] = 0.50
    assert failed(c)


def test_low_precision_is_blocked():
    c = copy.deepcopy(GOOD)
    c["test"]["precision"] = 0.18
    assert failed(c)


def test_large_f1_drop_is_blocked():
    c = copy.deepcopy(GOOD)
    c["test"]["f1"] -= 0.12
    assert failed(c)


def test_small_f1_wobble_is_allowed():
    c = copy.deepcopy(GOOD)
    c["test"]["f1"] -= 0.02
    assert not failed(c)


def test_month_to_month_collapse_is_blocked():
    c = copy.deepcopy(GOOD)
    c["slices"]["month_to_month"]["recall"] = 0.55
    assert failed(c)


def test_long_contract_collapse_is_blocked():
    c = copy.deepcopy(GOOD)
    c["slices"]["long_contract"]["recall"] = 0.05
    assert failed(c)


def test_small_slice_is_explicitly_skipped():
    c = copy.deepcopy(GOOD)
    c["slices"]["long_contract"].update(recall=0, n_positive=4)
    assert not failed(c)
    assert any("skipped, only 4" in required for _, _, _, required in quality_gate.evaluate(c, BASELINE))


def test_slice_at_minimum_size_is_enforced():
    c = copy.deepcopy(GOOD)
    c["slices"]["long_contract"].update(recall=0, n_positive=20)
    assert failed(c)
