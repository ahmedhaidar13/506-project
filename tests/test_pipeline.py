# A few checks on the parts of the pipeline we rely on most.
# They use small made-up data, so they run without data/ (and on GitHub Actions).
import json

import pandas as pd

import config
import model
import stage2_backtest as s2


def make_reports(years):
    # one tiny fake report per year, with the columns build_features expects
    return pd.DataFrame({
        "year": years,
        "label": [i % 2 for i in range(len(years))],
        "product_code": ["LZG", "QBJ"] * (len(years) // 2) + ["LZG"] * (len(years) % 2),
        "device_operator": "LAY USER/PATIENT",
        "device_class": "2",
        "report_source_code": "Manufacturer report",
        "device_manufacturer": "ACME",
        "brand_name": "PUMP X",
        "product_problems": "Pumping Stopped|Insufficient Device Problem Information",
        "part_battery": "0",
        "event_type": "Injury",
    })


def test_split_trains_on_past_and_tests_on_future():
    df = make_reports([2020, 2021, 2022, 2023, 2024, 2025])
    train, test = model.split(df)
    assert set(train["year"]) == set(config.TRAIN_YEARS)
    assert set(test["year"]) == set(config.TEST_YEARS)
    assert train["year"].max() < test["year"].min()


def test_features_line_up_and_leave_out_the_label(monkeypatch):
    monkeypatch.setattr(model, "cat_cols", model.feature_groups["original"] + ["part_battery"])
    train = make_reports([2020, 2021, 2022, 2023])
    test = make_reports([2024, 2025])
    test["device_operator"] = "HEALTH PROFESSIONAL"  # a category train never saw
    X_train, X_test = model.build_features(train, test)

    assert list(X_train.columns) == list(X_test.columns)
    assert not any(c.startswith(("label", "event_type")) for c in X_train.columns)
    # problem codes we drop for leakage must not become features
    for p in model.drop_problems:
        assert "problem_" + p not in X_train.columns
    assert "problem_Pumping Stopped" in X_train.columns


def test_keep_all_problems_brings_back_the_dropped_codes(monkeypatch):
    monkeypatch.setattr(model, "cat_cols", model.feature_groups["original"])
    monkeypatch.setattr(model, "drop_problems", [])
    train = make_reports([2020, 2021, 2022, 2023])
    train.loc[train.index[0], "product_problems"] = "Adverse Event Without Identified Device or Use Problem"
    X_train, _ = model.build_features(train, make_reports([2024, 2025]))
    assert "problem_Adverse Event Without Identified Device or Use Problem" in X_train.columns


def test_spike_detector_flags_big_jump_but_not_tiny_one():
    months = pd.period_range("2020-01", periods=20, freq="M")
    counts = pd.Series([3] * 20, index=months)
    counts.iloc[15] = 30   # big spike
    counts.iloc[10] = 4    # small bump, under MIN_COUNT
    flags = s2.flag_spikes(counts, months[0])
    assert flags.iloc[15]
    assert not flags.iloc[10]
    assert flags.sum() == 1


def test_repeat_recalls_within_12_months_are_not_new(tmp_path, monkeypatch):
    recalls = [
        {"product_code": "LZG", "res_event_number": "A", "event_date_initiated": "20210115"},
        {"product_code": "LZG", "res_event_number": "A", "event_date_initiated": "20210115"},  # same recall listed twice
        {"product_code": "LZG", "res_event_number": "B", "event_date_initiated": "20210610"},  # 5 months later: repeat
        {"product_code": "LZG", "res_event_number": "C", "event_date_initiated": "20220920"},  # 15 months after B: new
        {"product_code": "ZZZ", "res_event_number": "D", "event_date_initiated": "20220101"},  # not one of our devices
    ]
    (tmp_path / "recall.jsonl").write_text("\n".join(json.dumps(r) for r in recalls))
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    new = s2.load_new_recalls()
    assert sorted(new["recall_id"]) == ["A", "C"]