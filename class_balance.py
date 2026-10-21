import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

#load the cleaned data from clean_data.py
df = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv")
names = {0: "Malfunction", 1: "Injury/death"}

#count each class by year and by product code
def class_counts(col):
    counts = pd.crosstab(df[col], df["label"]).reindex(columns=[0, 1], fill_value=0)
    counts = counts.rename(columns=names)
    counts["total"] = counts.sum(axis=1)
    counts["injury_death_share"] = (counts["Injury/death"] / counts["total"]).round(4)
    return counts

by_year = class_counts("year")
by_code = class_counts("product_code")
print(by_year, "\n")
print(by_code, "\n")

#chart: share of each class, with report counts on top of each bar
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, counts, title in [(axes[0], by_year, "By year"), (axes[1], by_code, "By product code")]:
    shares = counts[["Malfunction", "Injury/death"]].div(counts["total"], axis=0)
    shares.plot(kind="bar", stacked=True, ax=ax, color=["#9db4c8", "#c0504d"], legend=False, rot=0)
    for i, (n, s) in enumerate(zip(counts["total"], counts["injury_death_share"])):
        ax.text(i, 1.01, f"{s:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, pad=30)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Share of reports")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc="upper right")
fig.suptitle("Class balance: injury/death vs malfunction (labels show injury/death share)")
fig.tight_layout()

fig_dir = config.ROOT / "figures"
fig_dir.mkdir(exist_ok=True)
fig.savefig(fig_dir / "class_balance.png", dpi=150)
print("chart saved:", fig_dir / "class_balance.png")

#baseline F1 on the test years: always predicting injury/death gives
#precision = p and recall = 1, so F1 = 2p / (1 + p)
test = df[df["year"].isin(config.TEST_YEARS)]
p = test["label"].mean()
print(f"\ntest rows (2024-25): {len(test):,}")
print(f"injury/death share p: {p:.4f}")
print(f"baseline F1 = 2p/(1+p): {2 * p / (1 + p):.4f}")
