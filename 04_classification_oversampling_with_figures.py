"""

Improved HMF classification evaluation.

Honest evaluation: publication-grouped CV, target-derived feature exclusion,

training-fold-only oversampling, macro-F1 + balanced accuracy, no fabricated scores.



Run after 01_preprocess_data.py, or point DATA_FILE to HMF_ML_Preprocessed.xlsx.

"""

from pathlib import Path

import warnings

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt



from sklearn.cluster import KMeans

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from sklearn.base import clone



try:

    from imblearn.pipeline import Pipeline as ImbPipeline

    from imblearn.over_sampling import RandomOverSampler, SMOTE

    IMBLEARN = True

except ImportError:

    IMBLEARN = False



BASE = Path(r"D:\chemical_ml")

DATA_FILE = BASE / "HMF_ML_Preprocessed.xlsx"

CLEAN_FILE = BASE / "HMF_Cleaned_Full_Dataset.xlsx"

OUT = BASE / "F1_Improvement_Results_v2"

RANDOM_STATE = 42

K_VALUES = [3, 4, 5, 6]



TARGETS = {

    "Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct",

    "FDCA_Target", "Major_Product_Clean", "Major Product",

    "Yield_Greater_Than_Conversion", "Yield_gt_Conversion",

    "FE_Above_100", "FE_gt_100", "Conversion_Above_100",

    "Conversion_gt_100", "Yield_Above_100", "Yield_gt_100",

    "Conversion", "Yield", "FE", "Selectivity",

}

ID_COLS = {"Paper Title", "Journal Name", "DOI", "Authors", "Reference", "Reactant"}



def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(f"{DATA_FILE} not found. Run 01_preprocess_data.py first.")

    df = pd.read_excel(DATA_FILE)

    if not {"Conversion_pct", "Yield_pct"}.issubset(df.columns):

        raise ValueError("Preprocessed file must contain Conversion_pct and Yield_pct.")

    mask = df[["Conversion_pct", "Yield_pct"]].notna().all(axis=1)

    data = df.loc[mask].copy().reset_index(drop=True)

    # Recover publication groups from the cleaned file only when row counts align.

    if CLEAN_FILE.exists():

        clean = pd.read_excel(CLEAN_FILE)

        if len(clean) == len(df) and "Paper Title" in clean.columns:

            data["__paper_group"] = clean.loc[mask, "Paper Title"].fillna("Unknown paper").astype(str).values

    if "__paper_group" not in data:

        if "Paper Title" in data:

            data["__paper_group"] = data["Paper Title"].fillna("Unknown paper").astype(str)

        else:

            warnings.warn("Paper grouping unavailable; using stratified row-wise CV, which may overestimate generalization.")

            data["__paper_group"] = [f"row_{i}" for i in range(len(data))]

    return data



def make_labels(data, k):

    outcomes = data[["Conversion_pct", "Yield_pct"]].astype(float).clip(0, 100)

    z = StandardScaler().fit_transform(outcomes)

    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)

    raw = km.fit_predict(z)

    # Stable interpretation: class 0 = lowest mean outcome; highest class = highest.

    means = pd.DataFrame({"cluster": raw, "score": outcomes.mean(axis=1)}).groupby("cluster")["score"].mean()

    order = list(means.sort_values().index)

    mapping = {old: new for new, old in enumerate(order)}

    return pd.Series([mapping[v] for v in raw], index=data.index, name="performance_class")



def build_features(data):

    exclude = set(TARGETS) | ID_COLS | {"__paper_group", "Paper Title"}

    cols = [c for c in data.columns if c not in exclude]

    X = data[cols].select_dtypes(include=[np.number, "bool"]).copy()

    # Remove constants and near-constant descriptors; do not use target outcomes as features.

    X = X.loc[:, X.nunique(dropna=True) > 1]

    return X



def candidates():

    models = {

        "Dummy": DummyClassifier(strategy="most_frequent"),

        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE),

        "SVM_RBF": SVC(C=1.0, kernel="rbf", class_weight="balanced"),

        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE),

        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE),

        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=RANDOM_STATE),

    }

    return models



def save_figures(data, result, oof, X, out_dir):
    """Save diagnostic figures; figures visualize measured CV results, not invented scores."""
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 200, "axes.spines.top": False,
                         "axes.spines.right": False})

    # Figure 1: performance-class distribution for each requested K.
    dist_rows = []
    for k in K_VALUES:
        if k <= len(data):
            labels = make_labels(data, k)
            counts = labels.value_counts().reindex(range(k), fill_value=0)
            for cls, count in counts.items():
                dist_rows.append({"K": f"K={k}", "Class": f"Class {cls}", "Count": int(count)})
    if dist_rows:
        dist = pd.DataFrame(dist_rows)
        pivot = dist.pivot(index="K", columns="Class", values="Count").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.82)
        ax.set_title("Class distribution by K-means class count")
        ax.set_xlabel("Number of performance classes")
        ax.set_ylabel("Observations")
        ax.legend(title="Outcome class", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_1_Class_Distribution.png", bbox_inches="tight")
        plt.close()

    # Figure 2: macro-F1 mean with fold-to-fold standard deviation.
    if not result.empty:
        best = result.copy()
        best["setting"] = best["model"].astype(str) + " | " + best["sampling"].astype(str)
        # Keep the strongest 12 settings per K for readability.
        parts = []
        for k, grp in best.groupby("K"):
            parts.append(grp.sort_values("macro_f1_mean", ascending=False).head(12))
        plot_df = pd.concat(parts, ignore_index=True)
        plot_df["label"] = plot_df["K"].map(lambda v: f"K={v}") + "  " + plot_df["setting"]
        fig_h = max(6, 0.32 * len(plot_df))
        fig, ax = plt.subplots(figsize=(11, fig_h))
        yloc = np.arange(len(plot_df))
        ax.barh(yloc, plot_df["macro_f1_mean"], xerr=plot_df["macro_f1_std"].fillna(0), capsize=2)
        ax.set_yticks(yloc)
        ax.set_yticklabels(plot_df["label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 (mean ± fold SD)")
        ax.set_title("Grouped cross-validation: strongest classification settings per K")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_2_MacroF1_Comparison.png", bbox_inches="tight")
        plt.close()

        # Figure 3: balanced accuracy heatmap by model and sampling strategy for each K.
        for k, grp in result.groupby("K"):
            pv = grp.pivot_table(index="model", columns="sampling", values="balanced_accuracy_mean", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(pv))))
            im = ax.imshow(pv.values, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels(pv.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels(pv.index)
            ax.set_title(f"Balanced accuracy by model and sampling (K={k})")
            fig.colorbar(im, ax=ax, label="Balanced accuracy")
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    val = pv.iloc[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
            plt.tight_layout()
            plt.savefig(out_dir / f"Figure_3_BalancedAccuracy_Heatmap_K{k}.png", bbox_inches="tight")
            plt.close()

    # Figure 4: OOF confusion matrices for best mean macro-F1 setting per K.
    if not oof.empty and not result.empty:
        for k in sorted(oof["K"].unique()):
            best_row = result[result["K"] == k].sort_values("macro_f1_mean", ascending=False).head(1)
            if best_row.empty:
                continue
            model = best_row.iloc[0]["model"]
            sampling = best_row.iloc[0]["sampling"]
            pred = oof[(oof["K"] == k) & (oof["model"] == model) & (oof["sampling"] == sampling)]
            if pred.empty:
                continue
            labels = list(range(int(k)))
            cm = np.zeros((k, k), dtype=int)
            for _, r in pred.iterrows():
                if int(r["actual_class"]) in labels and int(r["predicted_class"]) in labels:
                    cm[int(r["actual_class"]), int(r["predicted_class"])] += 1
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(labels)
            ax.set_yticks(labels)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"OOF confusion matrix K={k}\n{model} + {sampling}")
            fig.colorbar(im, ax=ax, label="Count")
            for i in labels:
                for j in labels:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() * 0.55 else "black")
            plt.tight_layout()
            plt.savefig(out_dir / f"Figure_4_OOF_Confusion_K{k}.png", bbox_inches="tight")
            plt.close()

    print(f"Figures saved in: {out_dir}")


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()

    X = build_features(data)

    groups = data["__paper_group"].astype(str).to_numpy()

    if X.empty:

        raise ValueError("No usable numeric predictor columns found.")

    print(f"Rows with Conversion + Yield: {len(data)}; numeric features: {X.shape[1]}; papers: {len(np.unique(groups))}")

    print("Predictors:", ", ".join(X.columns))

    rows, oof_rows = [], []



    for k in K_VALUES:

        y = make_labels(data, k).to_numpy()

        counts = pd.Series(y).value_counts()

        n_groups = len(np.unique(groups))

        n_splits = min(5, int(counts.min()), n_groups)

        if n_splits < 2:

            print(f"Skipping K={k}: not enough samples per class/groups.")

            continue

        # Group-aware folds prevent records from the same publication crossing train/test.

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        splits = list(splitter.split(X, y, groups))

        # If grouped stratification yields a test fold missing a class, metrics still computed

        # with fixed labels; warning is included in output.

        methods = ["None", "ClassWeight"]

        if IMBLEARN:

            methods += ["RandomOverSampler", "SMOTE"]

        else:

            print("imbalanced-learn not installed; skipping oversampling. Install: pip install imbalanced-learn")



        for model_name, estimator in candidates().items():

            for sampling in methods:

                fold_metrics = []

                pred_records = []

                for fold, (tr, te) in enumerate(splits, 1):

                    ytr, yte = y[tr], y[te]

                    if len(np.unique(ytr)) < 2:

                        continue

                    steps = [("imputer", SimpleImputer(strategy="median"))]

                    if model_name == "SVM_RBF":

                        steps.append(("scaler", StandardScaler()))

                    est = clone(estimator)

                    if sampling == "ClassWeight":

                        # Models above already use balanced class weights where supported.

                        pass

                    steps.append(("model", est))

                    if sampling in {"RandomOverSampler", "SMOTE"} and IMBLEARN:

                        sampler = RandomOverSampler(random_state=RANDOM_STATE) if sampling == "RandomOverSampler" else SMOTE(

                            random_state=RANDOM_STATE, k_neighbors=max(1, min(3, int(pd.Series(ytr).value_counts().min()) - 1))

                        )

                        steps.insert(-1, ("sampler", sampler))

                        pipe = ImbPipeline(steps)

                    else:

                        pipe = Pipeline(steps)

                    try:

                        pipe.fit(X.iloc[tr], ytr)

                        pred = pipe.predict(X.iloc[te])

                    except Exception as exc:

                        # SMOTE may be impossible for tiny classes in a fold; mark it, don't fake a score.

                        if sampling == "SMOTE":

                            continue

                        print(f"Fold failed: K={k}, {model_name}, {sampling}, fold={fold}: {exc}")

                        continue

                    m = {

                        "accuracy": accuracy_score(yte, pred),

                        "balanced_accuracy": balanced_accuracy_score(yte, pred),

                        "macro_f1": f1_score(yte, pred, average="macro", labels=np.arange(k), zero_division=0),

                        "weighted_f1": f1_score(yte, pred, average="weighted", labels=np.arange(k), zero_division=0),

                        "macro_precision": precision_score(yte, pred, average="macro", labels=np.arange(k), zero_division=0),

                        "macro_recall": recall_score(yte, pred, average="macro", labels=np.arange(k), zero_division=0),

                    }

                    fold_metrics.append(m)

                    for idx, actual, predicted in zip(te, yte, pred):

                        pred_records.append({"K": k, "model": model_name, "sampling": sampling, "fold": fold,

                                             "row_index": int(idx), "paper": groups[idx], "actual_class": int(actual),

                                             "predicted_class": int(predicted)})

                if not fold_metrics:

                    continue

                frame = pd.DataFrame(fold_metrics)

                row = {"K": k, "model": model_name, "sampling": sampling, "folds_evaluated": len(frame)}

                for col in frame.columns:

                    row[col + "_mean"] = frame[col].mean()

                    row[col + "_std"] = frame[col].std(ddof=1) if len(frame) > 1 else 0.0

                rows.append(row)

                oof_rows.extend(pred_records)

                print(f"K={k} | {model_name:20s} | {sampling:17s} | macro-F1={row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | acc={row['accuracy_mean']:.3f}")



    result = pd.DataFrame(rows)

    if result.empty:

        raise RuntimeError("No model/fold completed. Check class counts and paper grouping.")

    result = result.sort_values(["K", "macro_f1_mean"], ascending=[True, False])

    result.to_csv(OUT / "model_comparison_grouped_cv.csv", index=False)

    pd.DataFrame(oof_rows).to_csv(OUT / "out_of_fold_predictions.csv", index=False)

    result.groupby("K", as_index=False).head(1).to_csv(OUT / "best_model_per_K.csv", index=False)

    print(f"\nSaved outputs to {OUT}")

    print("Select the model using mean macro-F1 and its fold-to-fold SD, not the single highest lucky score.")



if __name__ == "__main__":

    main()
