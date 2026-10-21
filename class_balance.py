import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

names = {0: "Malfunction", 1: "Injury/death"}

# product codes with fewer reports than this are too small to show a reliable share
MIN_REPORTS = 100


#count each class by a column (year or product code)
def class_counts(df, col):
    counts = pd.crosstab(df[col], df["label"]).reindex(columns=[0, 1], fill_value=0)
    counts = counts.rename(columns=names)
    counts["total"] = counts.sum(axis=1)
    counts["injury_death_share"] = (counts["Injury/death"] / counts["total"]).round(4)
    return counts


#bar chart of the injury/death share, with the share and report count above each bar
def plot_shares(ax, counts, title):
    shares = counts["injury_death_share"]
    ax.bar(range(len(shares)), shares, color="#c0504d")
    for i, (s, n) in enumerate(zip(shares, counts["total"])):
        ax.text(i, s, f"{s:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(shares)))
    ax.set_xticklabels(counts.index, fontsize=8)
    ax.set_title(title)


if __name__ == "__main__":
    #load the cleaned data from clean_data.py
    df = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv")

    by_year = class_counts(df, "year")
    by_code = class_counts(df, "product_code")
    print(by_year, "\n")
    print(by_code, "\n")

    # Leave tiny product codes off the chart (they're still in the printed table)
    small = by_code[by_code["total"] < MIN_REPORTS]
    if len(small):
        print(f"not charted (under {MIN_REPORTS} reports): {', '.join(small.index)}\n")
    by_code = by_code[by_code["total"] >= MIN_REPORTS].sort_values("total", ascending=False)
    by_code.index = [c + "\n" + textwrap.fill(config.PRODUCT_CODES.get(c, ""), 14) for c in by_code.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True,
                             gridspec_kw={"width_ratios": [len(by_year), len(by_code)]})
    plot_shares(axes[0], by_year, "By year")
    plot_shares(axes[1], by_code, "By product code")
    top = max(by_year["injury_death_share"].max(), by_code["injury_death_share"].max())
    axes[0].set_ylim(0, top * 1.25)  # leave room for the labels
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Injury/death share of reports")
    fig.suptitle("Share of reports that are injury/death (the rest are malfunctions)")
    fig.tight_layout()

    fig_dir = config.ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "class_balance.png", dpi=150)
    print("chart saved:", fig_dir / "class_balance.png")

    #Baseline F1 on the test years: always predicting injury/death gives
    # precision = p and recall = 1, so F1 = 2p / (1 + p)
    test = df[df["year"].isin(config.TEST_YEARS)]
    p = test["label"].mean()
    print(f"\ntest rows (2024-25): {len(test):,}")
    print(f"injury/death share p: {p:.4f}")
    print(f"baseline F1 = 2p/(1+p): {2 * p / (1 + p):.4f}")