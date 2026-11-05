# Stage 1: decision tree on the structured fields only (no event description text yet)
# Saves the confusion matrix and feature importance plots to figures/.

#imports
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import argparse

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

import config

fig_dir = config.ROOT / "figures"

# one-hot encode these columns, in groups so we can add them one round at a time
#   python model.py                          all groups
#   python model.py --groups original        just what we had before
#   python model.py --groups original setting usage
feature_groups = {
    "original": ["product_code", "device_operator", "device_class", "report_source_code"],
    "setting": ["event_location", "reporter_occupation_code", "health_professional",
                "implant_flag", "serviced_by_3rd_party"],
    "usage": ["previous_use_code", "single_use_flag", "reprocessed_and_reused_flag"],
    "patient": ["age_group", "patient_sex"],
    "parts": ["part_battery", "part_tubing", "part_cartridge", "part_sensor", "part_transmitter",
              "part_adhesive", "part_inserter", "part_screen_buttons", "part_software_app",
              "part_receiver"],
    # brand is handled like manufacturer (top brands, the rest are "Other"), see build_features
    "brand": [],
}
# our main model: the best F1 in our comparison, and every feature makes sense
default_groups = ["original", "setting", "parts"]

# insulin pump product codes, for --pumps (looks only inside pumps, without the sensors)
pump_codes = ["LZG", "QFG", "QJI"]

cat_cols = []  # filled in from --groups when the script runs
fig_suffix = ""  # "_pumps" when running with --pumps, so plots don't overwrite each other
use_brand = False  # set to True when "brand" is in --groups

# manufacturer has a lot of values, so keep the most common ones and call the rest "Other"
n_manufacturers = 15

# same idea for brand names (the specific pump or monitor, like a certain pump model)
n_brands = 20

# product_problems can list several problems per report, keep the most common ones as 0/1 columns
n_problems = 30

# problems to leave out if they turn out to give away the label (leakage check)
# Something like "Adverse Event Without Identified Device or Use Problem"
drop_problems = ["Adverse Event Without Identified Device or Use Problem",
                 "Insufficient Device Problem Information"]
max_depth = 6


def load():
    df = pd.read_csv(config.PROCESSED_DIR / "events_clean.csv", dtype=str, keep_default_na=False)
    df["year"] = df["year"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


#split by year so we train on the past and test on the future
def split(df):
    train = df[df["year"].isin(config.TRAIN_YEARS)].copy()
    test = df[df["year"].isin(config.TEST_YEARS)].copy()
    return train, test


# turn the cleaned columns into numeric features
# the top manufacturers/problems are picked from the training set only so the test set doesn't leak in
def build_features(train, test):
    top_mfr = train["device_manufacturer"].value_counts().head(n_manufacturers).index

    # brand names are typed by reporters, so clean up case and spaces before counting
    def clean_brand(df):
        return df["brand_name"].str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    top_brands = clean_brand(train).value_counts().head(n_brands).index
    problem_counts = train["product_problems"].str.split("|").explode().value_counts()
    problem_counts = problem_counts[(problem_counts.index != "") & ~problem_counts.index.isin(drop_problems)]
    top_problems = problem_counts.head(n_problems).index

    def features(df):
        X = df[cat_cols].replace("", "Unknown").copy()
        X["manufacturer"] = df["device_manufacturer"].where(df["device_manufacturer"].isin(top_mfr), "Other")
        if use_brand:
            brand = clean_brand(df)
            X["brand"] = brand.where(brand.isin(top_brands), "Other")
        X = pd.get_dummies(X, dtype=int)
        problems = df["product_problems"].str.split("|")
        for p in top_problems:
            X["problem_" + p] = problems.apply(lambda ps: int(p in ps))
        return X

    X_train = features(train)
    # test can have categories train never saw, line its columns up with train's
    X_test = features(test).reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def plot_confusion_matrix(y_test, pred):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Malfunction", "Injury/Death"], cmap="Blues", colorbar=False
    )
    disp.ax_.set_title("Decision tree, 2024-2025 test set" + (" (pumps only)" if fig_suffix else ""))
    disp.figure_.tight_layout()
    disp.figure_.savefig(fig_dir / f"tree_confusion_matrix{fig_suffix}.png", dpi=150)
    plt.close(disp.figure_)

# Plot the top features, which can help us check for leakage
def plot_importances(importances):
    top = importances.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 15 features, decision tree" + (" (pumps only)" if fig_suffix else ""))
    fig.tight_layout()
    fig.savefig(fig_dir / f"tree_feature_importance{fig_suffix}.png", dpi=150)
    plt.close(fig)

# run the model and save plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=default_groups,
                        choices=list(feature_groups))
    parser.add_argument("--pumps", action="store_true", help="only insulin pump reports")
    args = parser.parse_args()
    for g in args.groups:
        cat_cols += feature_groups[g]
    use_brand = "brand" in args.groups
    print("feature groups:", ", ".join(args.groups))

    df = load()
    if args.pumps:
        df = df[df["product_code"].isin(pump_codes)]
        fig_suffix = "_pumps"
        print("pumps only:", len(df), "reports")
    train, test = split(df)
    print(f"train: {len(train)} rows, injury/death share {train['label'].mean():.3f}")
    print(f"test:  {len(test)} rows, injury/death share {test['label'].mean():.3f}")

    X_train, X_test = build_features(train, test)
    y_train, y_test = train["label"], test["label"]
    print(f"{X_train.shape[1]} features")

    # balanced class weights so the tree doesn't just predict malfunction every time
    tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=config.SEED)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)

    # compare to always predicting injury/death (the baseline from our proposal)
    baseline_f1 = f1_score(y_test, [1] * len(y_test))
    tree_f1 = f1_score(y_test, pred)
    print(f"\nbaseline injury/death F1: {baseline_f1:.3f}")
    print(f"tree injury/death F1:     {tree_f1:.3f}  (goal is baseline + 0.15 = {baseline_f1 + 0.15:.3f})")
    print(classification_report(y_test, pred, target_names=["Malfunction", "Injury/Death"]))

    # show the top features, which can help us check for leakage
    importances = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("top 10 features (if one problem_ column dominates, check it for leakage):")
    print(importances.head(10).round(3).to_string())

    # save plots
    fig_dir.mkdir(exist_ok=True)
    plot_confusion_matrix(y_test, pred)
    plot_importances(importances)
    print(f"\nsaved plots to {fig_dir.name}/")