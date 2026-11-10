# Clusters the event descriptions with K-Means to find common failure types, then
# checks how often each cluster ends in injury/death. Fit on 2020-2023 only so the
# test years don't leak in.

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

import config

fig_dir = config.ROOT / "figures"

pump_codes = ["LZG", "QFG", "QJI"]

n_words = 5000        #How many words the tfidf keeps
n_components = 100    # size after SVD, keeps K-Means fast and less noisy
n_top_terms = 6       # words shown per cluster


def load(pumps):
    cols = ["year", "label", "product_code", "event_description"]
    df = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv", usecols=cols, dtype=str,
                     keep_default_na=False)
    df["year"] = df["year"].astype(int)
    df["label"] = df["label"].astype(int)
    if pumps:
        df = df[df["product_code"].isin(pump_codes)]
    # reports with no description can't be clustered by text
    df = df[df["event_description"].str.strip() != ""]
    return df


# tfidf -> SVD -> normalize, all fit on the training years only
def build_text_features(train, test):
    # letters only (drops things like (b)(6)), words that show up in most reports are dropped
    tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z]{3,}", max_features=n_words,
                            min_df=5, max_df=0.4)
    X_train = tfidf.fit_transform(train["event_description"].str.lower())
    X_test = tfidf.transform(test["event_description"].str.lower())
    svd = TruncatedSVD(n_components=n_components, random_state=config.SEED)
    norm = Normalizer(copy=False)
    Z_train = norm.fit_transform(svd.fit_transform(X_train))
    Z_test = norm.transform(svd.transform(X_test))
    return tfidf, svd, Z_train, Z_test


# silhouette on a sample (it is slow on the full set)
def sweep(Z):
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
    print("k   silhouette")
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=5, random_state=config.SEED).fit(Z)
        print(f"{k:<3} {silhouette_score(Z[idx], km.labels_[idx]):.3f}")


# the words that describe each cluster: map the center back to word space
def top_terms(km, svd, tfidf):
    words = np.array(tfidf.get_feature_names_out())
    weights = km.cluster_centers_ @ svd.components_
    return [", ".join(words[np.argsort(row)[::-1][:n_top_terms]]) for row in weights]


def summarize(train, test, k, terms):
    rows = []
    for c in range(k):
        tr, te = train[train["cluster"] == c], test[test["cluster"] == c]
        both = pd.concat([tr, te])
        rows.append({
            "cluster": c,
            "top_words": terms[c],
            "train_reports": len(tr),
            "train_injury_death_share": tr["label"].mean() if len(tr) else np.nan,
            "test_reports": len(te),
            "test_injury_death_share": te["label"].mean() if len(te) else np.nan,
            "most_common_product_code": both["product_code"].mode().iat[0] if len(both) else "",
        })
    return pd.DataFrame(rows).sort_values("train_injury_death_share", ascending=False)


def plot_clusters(table, overall, suffix):
    t = table.iloc[::-1]  # highest share at the top of the chart
    labels = [f"C{c}: {', '.join(w.split(', ')[:3])}" for c, w in zip(t["cluster"], t["top_words"])]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(t) + 1.8))
    ax.barh(labels, t["train_injury_death_share"], color="#4C72B0", label="train 2020-2023")
    ax.scatter(t["test_injury_death_share"], labels, color="#DD8452", zorder=3,
               label="test 2024-2025")
    ax.axvline(overall, color="gray", linestyle="--", linewidth=1, label=f"overall train ({overall:.2f})")
    ax.set_xlabel("Share of reports that are injury/death")
    ax.set_title("Injury/death share by text cluster" + (" (pumps only)" if suffix else ""))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"kmeans_clusters{suffix}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="number of clusters")
    parser.add_argument("--pumps", action="store_true", help="only insulin pump reports")
    parser.add_argument("--sweep", action="store_true", help="print silhouette scores for k = 4..12")
    args = parser.parse_args()

    df = load(args.pumps)
    suffix = "_pumps" if args.pumps else ""
    train = df[df["year"].isin(config.TRAIN_YEARS)].copy()
    test = df[df["year"].isin(config.TEST_YEARS)].copy()
    print(f"{'pumps only: ' if args.pumps else ''}train {len(train)} reports, test {len(test)} reports")

    tfidf, svd, Z_train, Z_test = build_text_features(train, test)
    if args.sweep:
        sweep(Z_train)
        raise SystemExit

    km = KMeans(n_clusters=args.k, n_init=10, random_state=config.SEED).fit(Z_train)
    train["cluster"] = km.labels_
    test["cluster"] = km.predict(Z_test)

    table = summarize(train, test, args.k, top_terms(km, svd, tfidf))
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(table.round(3).to_string(index=False))

    overall = train["label"].mean()
    print(f"\noverall train injury/death share: {overall:.3f}")
    fig_dir.mkdir(exist_ok=True)
    plot_clusters(table, overall, suffix)
    table.to_csv(config.PROCESSED_DIR / f"kmeans_clusters{suffix}.csv", index=False)
    print(f"saved figures/kmeans_clusters{suffix}.png and data/processed/kmeans_clusters{suffix}.csv")
