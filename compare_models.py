# Compares a decision tree, Naive Bayes and KNN using the same setup as model.py.
# Injury/death is rare, so we tune the cutoff (and k) on 2023 instead of using 0.5,
# then refit on 2020-2023 and test once on 2024-2025.
#
# usage: python compare_models.py [--pumps]

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import config
import model  # reuse load(), split(), build_features() and the feature groups

fig_dir = config.ROOT / "figures"
knn_ks = [5, 15, 50]                  # choices of k tried on the validation year
cutoffs = np.arange(0.05, 0.96, 0.05)  # probability cutoffs tried on the validation year


#Best cutoff (the one with the highest injury/death F1) for predicted probabilities
def best_cutoff(y_true, proba):
    scores = [f1_score(y_true, proba >= c, zero_division=0) for c in cutoffs]
    return cutoffs[int(np.argmax(scores))], max(scores)


# pick the cutoff (and k) on 2023, then refit on all training years and predict the test set
def tune_and_predict(make_model, params, train, test, name):
    fit_part = train[train["year"] < 2023]
    val_part = train[train["year"] == 2023]
    X_fit, X_val = model.build_features(fit_part, val_part)
    best = None
    for p in params:
        m = make_model(p).fit(X_fit, fit_part["label"])
        cutoff, score = best_cutoff(val_part["label"], m.predict_proba(X_val)[:, 1])
        if best is None or score > best[0]:
            best = (score, p, cutoff)
    _, p, cutoff = best
    print(f"{name}: picked {'k=' + str(p) if p else 'default settings'}, cutoff {cutoff:.2f} "
          f"(validation F1 {best[0]:.3f})")
    X_train, X_test = model.build_features(train, test)
    m = make_model(p).fit(X_train, train["label"])
    return (m.predict_proba(X_test)[:, 1] >= cutoff).astype(int), p, cutoff

#Run the main comparison
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pumps", action="store_true", help="only insulin pump reports")
    args = parser.parse_args()
    suffix = "_pumps" if args.pumps else ""

    # same features as our main decision tree
    for g in model.default_groups:
        model.cat_cols += model.feature_groups[g]
    df = model.load()
    if args.pumps:
        df = df[df["product_code"].isin(model.pump_codes)]
    train, test = model.split(df)
    y_test = test["label"]
    print(f"{'pumps only: ' if args.pumps else ''}train {len(train)}, test {len(test)}, "
          f"test injury/death share {y_test.mean():.3f}\n")

    X_train, X_test = model.build_features(train, test)
    tree = DecisionTreeClassifier(max_depth=model.max_depth, class_weight="balanced",
                                  random_state=config.SEED).fit(X_train, train["label"])
    preds = {"Decision tree": tree.predict(X_test)}
    preds["Naive Bayes"], _, _ = tune_and_predict(lambda p: BernoulliNB(), [None], train, test, "Naive Bayes")
    preds["KNN"], k, _ = tune_and_predict(lambda k: KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1),
                                          knn_ks, train, test, "KNN")

    baseline = f1_score(y_test, [1] * len(y_test))
    rows = [{"model": "Always injury/death (baseline)", "f1": baseline, "precision": y_test.mean(), "recall": 1.0}]
    for name, p in preds.items():
        rows.append({"model": name, "f1": f1_score(y_test, p), "precision": precision_score(y_test, p, zero_division=0),
                     "recall": recall_score(y_test, p)})
    table = pd.DataFrame(rows)
    table["beats_baseline_by"] = table["f1"] - baseline
    print(f"\ngoal: injury/death F1 of at least baseline + 0.15 = {baseline + 0.15:.3f}")
    print(table.round(3).to_string(index=False))
    table.to_csv(config.PROCESSED_DIR / f"model_comparison{suffix}.csv", index=False)

    # confusion matrices side by side
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4))
    for ax, (name, p) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, p, display_labels=["Malfunction", "Injury/Death"],
                                                cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(f"{name}\nF1 {f1_score(y_test, p):.2f}", fontsize=10)
    fig.suptitle("Test set 2024-2025" + (" (pumps only)" if args.pumps else ""))
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"model_comparison{suffix}.png", dpi=150)
    print(f"\nsaved figures/model_comparison{suffix}.png and data/processed/model_comparison{suffix}.csv")