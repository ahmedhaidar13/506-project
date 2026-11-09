# Stage 2: recall backtest
# Does a spike in injury/death reports show up months before a recall for the same product code?
#
# Needs data/raw/recall.jsonl (from data_collection.py) and data/processed/events_clean.csv (from clean_data.py).
# Saves data/processed/stage2_results.csv (summary) and stage2_recalls.csv (one row per recall).
#
# Note: events_clean.csv is a sample of the full MAUDE data (sampled in proportion to daily
# volume), so monthly counts are scaled down, but they are scaled the same way for every month.

import json

import numpy as np
import pandas as pd

import config

# How the spike detector works: a month is flagged if its count is far above the trailing average
WINDOW = 12        # months of history to compare against (current month not included)
MIN_HISTORY = 6    # need at least this many months of history before we flag anything
Z_THRESHOLD = 2.0  # how many standard deviations above the average counts as a spike
MIN_COUNT = 5      # ignore tiny spikes (a jump from 0 to 2 reports isn't a signal)

# how we score it against recalls
MIN_LEAD = 3       # a flag only counts as early warning if it comes at least this many months before
MAX_LEAD = 12      # ...and no more than this many months before (older flags are unrelated)
REPEAT_GAP = 12    # a recall is only "new" if the product code had no recall in the last 12 months


# monthly report counts per product code, as a table (rows = months, columns = product codes)
def monthly_counts(df, only_serious):
    if only_serious:
        df = df[df["label"] == 1]
    months = pd.period_range(f"{config.START_YEAR}-01", f"{config.END_YEAR}-12", freq="M")
    counts = (df.assign(month=df["date_received"].dt.to_period("M"))
                .groupby(["month", "product_code"]).size()
                .unstack(fill_value=0))
    return counts.reindex(index=months, fill_value=0)


# Flag months where the count is a spike compared to the trailing WINDOW months
# starts monitoring a code at its first month with any report in the total counts (start_month)
def flag_spikes(series, start_month):
    series = series[series.index >= start_month]
    history = series.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY)
    mean = history.mean()
    std = history.std().clip(lower=1)  # floor so a flat history doesn't make every bump a spike
    flagged = (series >= mean + Z_THRESHOLD * std) & (series >= MIN_COUNT)
    return flagged.fillna(False)


#Load recalls, keep ours, drop repeat recalls of the same product code within 12 months
def load_new_recalls():
    rows = []
    with open(config.RAW_DIR / "recall.jsonl") as f:
        for line in f:
            r = json.loads(line)
            date = r.get("event_date_initiated") or r.get("event_date_posted") or r.get("create_date")
            rows.append({"product_code": r.get("product_code"),
                         "recall_id": r.get("res_event_number"),
                         "date": date})
    rec = pd.DataFrame(rows)
    rec["date"] = pd.to_datetime(rec["date"].astype(str), errors="coerce")
    rec = rec.dropna(subset=["date"])
    rec = rec[rec["product_code"].isin(config.PRODUCT_CODES)]

    #One recall can be listed several times (once per product), so count each id once per code
    rec = rec.drop_duplicates(subset=["product_code", "recall_id"])
    rec = rec.sort_values(["product_code", "date"])
    rec["month"] = rec["date"].dt.to_period("M")

    # a recall is new only if the previous recall for this code (new or not) was 12+ months earlier
    gap = rec.groupby("product_code")["month"].diff().apply(lambda d: d.n if pd.notna(d) else np.inf)
    rec["is_new"] = gap >= REPEAT_GAP

    rec["in_window"] = rec["date"].dt.year.between(config.START_YEAR, config.END_YEAR)
    print("recalls per product code (in 2020-2025):")
    for code, name in config.PRODUCT_CODES.items():
        sub = rec[(rec["product_code"] == code) & rec["in_window"]]
        print(f"  {code} ({name}): {len(sub)} total, {int(sub['is_new'].sum())} new after the 12-month rule")
    return rec[rec["is_new"] & rec["in_window"]]


#Score one signal (a table of True/False flags per month and product code) against the new recalls
def score(flags, recalls, start_months, name):
    recall_rows = []
    for _, r in recalls.iterrows():
        code = r["product_code"]
        if code not in flags.columns:
            continue
        # can't judge a recall if we weren't monitoring this code long enough before it
        first_checkable = start_months[code] + WINDOW
        if r["month"] < first_checkable + MIN_LEAD:
            continue
        f = flags[code]
        flag_months = f[f].index
        leads = [(r["month"] - m).n for m in flag_months if MIN_LEAD <= (r["month"] - m).n <= MAX_LEAD]
        recall_rows.append({"signal": name, "product_code": code, "recall_id": r["recall_id"],
                            "recall_month": str(r["month"]),
                            "caught_early": len(leads) > 0,
                            "lead_months": min(leads) if leads else np.nan})  # nearest qualifying flag, so noise earlier in the window can't inflate it
    per_recall = pd.DataFrame(recall_rows)

    # false flags: a flag "episode" is a run of consecutive flagged months. It's false if none of its
    # months came MIN_LEAD to MAX_LEAD months before a new recall of the same code
    false_flags, device_months = 0, 0
    total_flags = 0
    for code in flags.columns:
        f = flags[code]
        device_months += int((f.index >= start_months[code] + WINDOW).sum())
        code_recalls = recalls[recalls["product_code"] == code]["month"].tolist()
        in_episode, episode_useful = False, False
        for month, flagged in f.items():
            if flagged:
                useful = any(MIN_LEAD <= (rm - month).n <= MAX_LEAD for rm in code_recalls)
                if not in_episode:
                    in_episode, episode_useful = True, False
                    total_flags += 1
                episode_useful = episode_useful or useful
            elif in_episode:
                false_flags += 0 if episode_useful else 1
                in_episode = False
        if in_episode:
            false_flags += 0 if episode_useful else 1

    n = len(per_recall)
    caught = per_recall["caught_early"].sum() if n else 0
    summary = {
        "signal": name,
        "recalls_evaluated": n,
        "recalls_caught_early": int(caught),
        "share_caught_early": round(caught / n, 3) if n else np.nan,
        "median_lead_months": per_recall["lead_months"].median() if n else np.nan,
        "flag_episodes": total_flags,
        "false_flags": false_flags,
        "false_flags_per_device_year": round(false_flags / (device_months / 12), 3) if device_months else np.nan,
    }
    return summary, per_recall

# run main
if __name__ == "__main__":
    events = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv", dtype={"label": int},
                         keep_default_na=False)
    events["date_received"] = pd.to_datetime(events["date_received"])

    total = monthly_counts(events, only_serious=False)    # baseline: all reports
    serious = monthly_counts(events, only_serious=True)   # our signal: actual injury/death reports
    # a code with no injury/death reports at all still needs a (zero) column
    serious = serious.reindex(columns=total.columns, fill_value=0)
    codes = [c for c in total.columns if total[c].sum() > 0]

    # start watching each code at its first month with any report
    start_months = {c: total[c][total[c] > 0].index[0] for c in codes}

    recalls = load_new_recalls()
    print(f"\nnew recalls to test against: {len(recalls)}")

    results, per_recall = [], []
    for name, counts in [("injury/death spike (our signal)", serious),
                         ("raw report spike (baseline)", total)]:
        # codes start being watched at different months, so months before a code's start are False
        flags = pd.DataFrame({c: flag_spikes(counts[c], start_months[c]).reindex(counts.index, fill_value=False)
                              for c in codes})
        summary, rows = score(flags, recalls, start_months, name)
        results.append(summary)
        per_recall.append(rows)

    results = pd.DataFrame(results)
    per_recall = pd.concat(per_recall, ignore_index=True)
    print("\n", results.T.to_string(header=False))
    print(f"\nStage 2 goal: catch at least 50% of recalls {MIN_LEAD}+ months early AND beat the baseline")

    # save results
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.PROCESSED_DIR / "stage2_results.csv", index=False)
    per_recall.to_csv(config.PROCESSED_DIR / "stage2_recalls.csv", index=False)
    print("saved: stage2_results.csv, stage2_recalls.csv")