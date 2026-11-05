# Answers our research question with simple comparisons:
# for each device, user, and setting factor, what share of reports involve injury or death?
#
#   python findings.py
#
# Makes charts for all devices and for insulin pumps only:
#   figures/findings_device.png, findings_user.png, findings_setting.png
#   figures/findings_device_pumps.png, ... (same charts, pumps only)
# and saves every number to data/processed/findings.csv
#
# These are shares among REPORTS, not risk per device or per patient.
# MAUDE has no count of how many people use each device, so word findings
# as "more common among injury reports", not "causes" or "riskier".

import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

fig_dir = config.ROOT / "figures"

device_types = {
    "LZG": "Insulin pump", "QFG": "Insulin pump", "QJI": "Insulin pump",
    "QBJ": "CGM sensor", "MDS": "CGM sensor",
    "NBW": "Blood glucose meter",
}

part_cols = ["part_battery", "part_tubing", "part_cartridge", "part_sensor", "part_transmitter",
             "part_adhesive", "part_inserter", "part_screen_buttons", "part_software_app",
             "part_receiver"]

# the three parts of our research question: {group: {chart title: column}}
factor_groups = {
    "device": {
        "Device type": "device_type",
        "Manufacturer": "device_manufacturer",
        "Problem code": "product_problems",   # several per report, handled separately
        "Part mentioned in description": "parts",  # from the part_ columns
        "Usage (first use / reuse)": "previous_use_code",
        "Labeled single use": "single_use_flag",
    },
    "user": {
        "Device operator": "device_operator",
        "Reporter occupation": "reporter_occupation_code",
        "Patient age group": "age_group",
        "Patient sex": "patient_sex",
    },
    "setting": {
        "Reporter is a health professional": "health_professional",
    },
}

# problem codes that basically restate the label (same list as drop_problems in model.py)
near_label_problems = ["Adverse Event Without Identified Device or Use Problem",
                       "Insufficient Device Problem Information"]

# readable names for raw MAUDE codes (FDA MDR data file code keys and Form 3500A)
code_labels = {
    "previous_use_code": {"I": "Initial use", "R": "Reuse", "U": "Unknown", "*": "Invalid data"},
    "reporter_occupation_code": {"003": "Non-healthcare professional"},
    "health_professional": {"Y": "Yes", "N": "No", "*": "Invalid data"},
    "single_use_flag": {"Y": "Yes", "N": "No"},
}

max_rows = 12  # most values shown per chart (the most common ones)


def load():
    df = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv", dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df["device_type"] = df["product_code"].map(device_types).fillna("Other")
    for c in part_cols:
        df[c] = df[c].astype(int)
    return df


# 95% confidence interval for a share (Wilson interval, works for small counts too)
def wilson(share, n, z=1.96):
    center = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


# injury/death share and report count for each value of one factor
def factor_table(df, col, min_n):
    if col == "product_problems":
        # a report can have several problems, so count it under each one
        long = df[["label", col]].assign(value=df[col].str.split("|")).explode("value")
        long = long[(long["value"] != "") & ~long["value"].isin(near_label_problems)]
    elif col == "parts":
        # share among reports that mention each part
        long = pd.concat([df.loc[df[c] == 1, ["label"]].assign(value=c.replace("part_", ""))
                          for c in part_cols])
    else:
        value = df[col].replace("", "Unknown").replace(code_labels.get(col, {}))
        long = df[["label"]].assign(value=value)

    t = long.groupby("value")["label"].agg(share="mean", n="count").reset_index()
    t = t[t["n"] >= min_n].sort_values("n", ascending=False).head(max_rows)
    t[["low", "high"]] = t.apply(lambda r: pd.Series(wilson(r["share"], r["n"])), axis=1)
    return t.sort_values("share")


def plot_group(df, group, suffix, min_n):
    overall = df["label"].mean()
    tables = {}
    for title, col in factor_groups[group].items():
        t = factor_table(df, col, min_n)
        if len(t) >= 2:  # nothing to compare with only one value
            tables[title] = t
    if not tables:
        return []

    heights = [len(t) + 1.5 for t in tables.values()]
    fig, axes = plt.subplots(len(tables), 1, figsize=(9, 0.32 * sum(heights) + 1),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (title, t) in zip(axes[:, 0], tables.items()):
        labels = [textwrap.shorten(str(v), 45) + f"  (n={n:,})" for v, n in zip(t["value"], t["n"])]
        err = [t["share"] - t["low"], t["high"] - t["share"]]
        ax.barh(labels, t["share"], xerr=err, color="#c0504d", ecolor="gray", capsize=2)
        ax.axvline(overall, color="black", linestyle="--", linewidth=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.tick_params(axis="y", labelsize=8)

    where = "insulin pumps only" if suffix else "all devices"
    fig.suptitle(f"Share of reports involving injury or death: {group} factors ({where})\n"
                 f"dashed line = overall share ({overall:.1%}), bars show 95% intervals",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"findings_{group}{suffix}.png", dpi=150)
    plt.close(fig)

    # keep the numbers for the README
    return [t.assign(group=group, factor=title, devices=where) for title, t in tables.items()]


if __name__ == "__main__":
    df = load()
    fig_dir.mkdir(exist_ok=True)
    pumps = df[df["device_type"] == "Insulin pump"]

    all_rows = []
    # pumps have fewer reports, so allow smaller groups there
    for data, suffix, min_n in [(df, "", 100), (pumps, "_pumps", 50)]:
        print(f"\n{'pumps only' if suffix else 'all devices'}: {len(data):,} reports, "
              f"injury/death share {data['label'].mean():.3f}")
        for group in factor_groups:
            rows = plot_group(data, group, suffix, min_n)
            all_rows += rows
            # print the highest and lowest share for each factor
            for t in rows:
                hi, lo = t.iloc[-1], t.iloc[0]
                print(f"  {t['factor'].iloc[0]}: highest {hi['value']} ({hi['share']:.1%}, n={hi['n']:,}), "
                      f"lowest {lo['value']} ({lo['share']:.1%}, n={lo['n']:,})")

    out = pd.concat(all_rows)[["devices", "group", "factor", "value", "n", "share", "low", "high"]]
    out.round(4).to_csv(config.PROCESSED_DIR / "findings.csv", index=False)
    print(f"\nsaved charts to {fig_dir.name}/ and numbers to findings.csv")