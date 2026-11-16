# Plots injury/death reports per month for each recalled product code, with our
# spike flags, the baseline flags, and recall dates. A flag in the pink window
# (3-12 months before a recall) counts as a catch. Reuses stage2_backtest.py.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import stage2_backtest as s2

fig_dir = config.ROOT / "figures"

if __name__ == "__main__":
    events = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv", dtype={"label": int},
                         keep_default_na=False)
    events["date_received"] = pd.to_datetime(events["date_received"])

    total = s2.monthly_counts(events, only_serious=False)
    serious = s2.monthly_counts(events, only_serious=True).reindex(columns=total.columns, fill_value=0)
    codes = [c for c in total.columns if total[c].sum() > 0]
    start_months = {c: total[c][total[c] > 0].index[0] for c in codes}
    recalls = s2.load_new_recalls()
    print(f"\nnew recalls: {len(recalls)}")

    # only draw product codes that have at least one new recall
    plot_codes = [c for c in codes if c in set(recalls["product_code"])]
    fig, axes = plt.subplots(len(plot_codes), 1, figsize=(11, 2.3 * len(plot_codes) + 1),
                             sharex=True, squeeze=False)

    for ax, code in zip(axes[:, 0], plot_codes):
        x = serious.index.to_timestamp()
        ax.plot(x, serious[code].values, color="#4C72B0", linewidth=1.3)

        ours = s2.flag_spikes(serious[code], start_months[code]).reindex(serious.index, fill_value=False)
        base = s2.flag_spikes(total[code], start_months[code]).reindex(serious.index, fill_value=False)
        ymax = max(serious[code].max(), 1)
        ax.scatter(x[ours.values], serious[code][ours].values, color="#DD8452", s=28, zorder=3)
        ax.scatter(x[base.values], [ymax * 1.12] * int(base.sum()), color="gray", marker="v",
                   s=22, zorder=3)

        for _, r in recalls[recalls["product_code"] == code].iterrows():
            ax.axvspan((r["month"] - s2.MAX_LEAD).to_timestamp(), (r["month"] - s2.MIN_LEAD).to_timestamp(),
                       color="#C44E52", alpha=0.12)
            ax.axvline(r["month"].to_timestamp(), color="#C44E52", linestyle="--", linewidth=1.2)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_ylabel("reports / month", fontsize=8)
        ax.set_title(f"{code}: {config.PRODUCT_CODES[code]}", fontsize=10, loc="left")

    # one legend for the whole figure
    handles = [plt.Line2D([], [], color="#4C72B0", label="injury/death reports per month"),
               plt.Line2D([], [], color="#DD8452", marker="o", linestyle="", label="our signal flagged"),
               plt.Line2D([], [], color="gray", marker="v", linestyle="", label="baseline (raw spike) flagged"),
               plt.Line2D([], [], color="#C44E52", linestyle="--", label="recall"),
               plt.Rectangle((0, 0), 1, 1, color="#C44E52", alpha=0.12, label="early-warning window (3-12 months before)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Stage 2: injury/death report signal vs. recall dates", fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "stage2_timeline.png", dpi=150)
    print("saved figures/stage2_timeline.png")
