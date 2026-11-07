"""
Combined learning-curve figure for HMF electrooxidation regression.

Creates ONE combined 4x4 figure (targets x models), similar to the supplied
reference image. It does not save separate learning-curve images or create
a new output folder. CSV results and the combined PNG/PDF are saved directly
inside D:\\chemical_ml.

Validation is grouped by publication where Paper Title is available.
Negative validation R² values are retained and are meaningful.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Change BASE only if your project is stored somewhere else.
BASE = Path(r"D:\chemical_ml")
DATA_FILE = BASE / "HMF_ML_Preprocessed.xlsx"
CLEAN_FILE = BASE / "HMF_Cleaned_Full_Dataset.xlsx"

TARGETS = ["Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct"]
MODEL_ORDER = ["Decision Tree", "SVR", "Random Forest", "Neural Network"]
EXCLUDE = set(TARGETS) | {
    "FDCA_Target", "Major_Product_Clean", "Major Product",
    "Yield_Greater_Than_Conversion", "Yield_gt_Conversion",
    "FE_Above_100", "FE_gt_100", "Conversion_Above_100",
    "Conversion_gt_100", "Yield_Above_100", "Yield_gt_100",
    "Paper Title", "Journal Name", "DOI", "Authors"
}
FRACTIONS = np.linspace(0.35, 1.0, 7)
RANDOM_STATE = 42


def make_model(name):
    if name == "Decision Tree":
        return DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=3, random_state=RANDOM_STATE
        )
    if name == "SVR":
        return SVR(C=3.0, epsilon=0.1, kernel="rbf")
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "Neural Network":
        return MLPRegressor(
            hidden_layer_sizes=(16, 8), activation="relu",
            # Adam is generally more practical than L-BFGS for this small,
            # noisy scientific dataset; predictors are scaled in make_pipeline().
            solver="adam", alpha=0.01, learning_rate_init=0.001,
            max_iter=5000, tol=1e-4, n_iter_no_change=40,
            random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in {"SVR", "Neural Network"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", make_model(name)))
    return Pipeline(steps)


def choose_group_subset(train_indices, groups, fraction, seed):
    """Choose whole publication groups, keeping the validation groups unseen."""
    train_groups = np.unique(groups[train_indices])
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(train_groups)
    n_groups = max(1, int(np.ceil(len(shuffled) * fraction)))
    chosen = set(shuffled[:n_groups].tolist())
    subset = np.array([i for i in train_indices if groups[i] in chosen], dtype=int)
    return subset


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Put HMF_ML_Preprocessed.xlsx in D:\\chemical_ml or update BASE."
        )

    df = pd.read_excel(DATA_FILE)

    # Prefer paper identifiers from the cleaned file only when row alignment is exact.
    groups = None
    if CLEAN_FILE.exists():
        clean = pd.read_excel(CLEAN_FILE)
        if len(clean) == len(df) and "Paper Title" in clean.columns:
            groups = clean["Paper Title"].fillna("Unknown paper").astype(str).to_numpy()

    if groups is None and "Paper Title" in df.columns:
        groups = df["Paper Title"].fillna("Unknown paper").astype(str).to_numpy()

    if groups is None:
        warnings.warn(
            "Paper Title was not found. Using row-wise CV; this can overestimate "
            "generalization to unseen publications."
        )
        groups = np.array([f"row_{i}" for i in range(len(df))], dtype=object)

    X = df[[c for c in df.columns if c not in EXCLUDE]].select_dtypes(
        include=["number", "bool"]
    ).copy()
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.empty:
        raise ValueError("No usable numeric predictor columns were found.")

    # Create one combined figure: 4 target rows x 4 model columns.
    fig, axes = plt.subplots(
        nrows=len(TARGETS), ncols=len(MODEL_ORDER),
        figsize=(18, 15), sharex=False, sharey=False
    )
    fig.suptitle(
        "Learning Curves for HMF Electrooxidation Performance Prediction",
        fontsize=17, fontweight="bold", y=0.995
    )

    curve_rows = []
    metric_rows = []
    panel_labels = list("abcdefghijklmnop")
    panel_index = 0

    for target_idx, target in enumerate(TARGETS):
        for model_idx, model_name in enumerate(MODEL_ORDER):
            ax = axes[target_idx, model_idx]
            ax.text(
                0.96, 0.90, f"({panel_labels[panel_index]})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold"
            )
            panel_index += 1

            if target not in df.columns:
                ax.set_visible(False)
                continue

            y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y_all)
            row_indices = np.flatnonzero(valid)
            y = y_all[valid]
            X_t = X.iloc[row_indices].reset_index(drop=True)
            g_t = groups[row_indices]

            if len(y) < 12:
                ax.text(0.5, 0.5, f"Insufficient data\nn={len(y)}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{target} — {model_name}")
                continue

            n_groups = len(np.unique(g_t))
            if n_groups >= 2:
                n_splits = min(5, n_groups)
                splits = list(GroupKFold(n_splits=n_splits).split(X_t, y, g_t))
            else:
                n_splits = min(5, len(y))
                splits = list(KFold(
                    n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
                ).split(X_t, y))

            x_sizes, train_means, val_means = [], [], []
            train_sds, val_sds = [], []

            # Overall grouped-CV metrics for this target/model.
            fold_r2, fold_mae, fold_rmse = [], [], []
            for train_idx, test_idx in splits:
                try:
                    pipe = make_pipeline(model_name)
                    pipe.fit(X_t.iloc[train_idx], y[train_idx])
                    pred = pipe.predict(X_t.iloc[test_idx])
                    if len(test_idx) > 1:
                        fold_r2.append(r2_score(y[test_idx], pred))
                    fold_mae.append(mean_absolute_error(y[test_idx], pred))
                    fold_rmse.append(mean_squared_error(y[test_idx], pred) ** 0.5)
                except Exception as exc:
                    warnings.warn(f"{target}/{model_name} metric fold skipped: {exc}")

            metric_rows.append({
                "target": target, "model": model_name, "n": len(y),
                "paper_groups": n_groups,
                "R2_mean": float(np.nanmean(fold_r2)) if fold_r2 else np.nan,
                "R2_sd": float(np.nanstd(fold_r2, ddof=1)) if len(fold_r2) > 1 else np.nan,
                "MAE_mean": float(np.mean(fold_mae)) if fold_mae else np.nan,
                "RMSE_mean": float(np.mean(fold_rmse)) if fold_rmse else np.nan,
                "folds": len(fold_mae)
            })

            for frac_idx, frac in enumerate(FRACTIONS):
                train_scores, val_scores, sizes = [], [], []
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    subset = choose_group_subset(
                        train_idx, g_t, frac,
                        RANDOM_STATE + 100 * frac_idx + fold_idx
                    )
                    if len(subset) < 4 or len(test_idx) < 2:
                        continue
                    try:
                        pipe = make_pipeline(model_name)
                        pipe.fit(X_t.iloc[subset], y[subset])
                        train_pred = pipe.predict(X_t.iloc[subset])
                        val_pred = pipe.predict(X_t.iloc[test_idx])
                        train_scores.append(r2_score(y[subset], train_pred))
                        val_scores.append(r2_score(y[test_idx], val_pred))
                        sizes.append(len(subset))
                    except Exception as exc:
                        warnings.warn(
                            f"{target}/{model_name}, fraction={frac:.2f} skipped: {exc}"
                        )

                if train_scores and val_scores:
                    x_sizes.append(float(np.mean(sizes)))
                    train_means.append(float(np.mean(train_scores)))
                    val_means.append(float(np.mean(val_scores)))
                    train_sds.append(float(np.std(train_scores, ddof=1)) if len(train_scores) > 1 else 0.0)
                    val_sds.append(float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0)
                    curve_rows.append({
                        "target": target, "model": model_name, "fraction": frac,
                        "train_n_mean": float(np.mean(sizes)),
                        "train_R2_mean": float(np.mean(train_scores)),
                        "train_R2_sd": float(np.std(train_scores, ddof=1)) if len(train_scores) > 1 else 0.0,
                        "validation_R2_mean": float(np.mean(val_scores)),
                        "validation_R2_sd": float(np.std(val_scores, ddof=1)) if len(val_scores) > 1 else 0.0,
                        "cv": "GroupKFold" if n_groups >= 2 else "KFold"
                    })

            if x_sizes:
                ax.plot(x_sizes, train_means, "o-", color="red",
                        linewidth=1.5, markersize=4, label="Training R²")
                ax.plot(x_sizes, val_means, "o-", color="green",
                        linewidth=1.5, markersize=4, label="Cross-validation R²")
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_title(
                    f"{target.replace('_pct', '').replace('_', ' ')} (%) — {model_name}",
                    fontsize=10, fontweight="bold"
                )
                ax.set_xlabel("Training examples", fontsize=8)
                ax.set_ylabel("R² score", fontsize=8)
                ax.tick_params(axis="both", labelsize=8)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, loc="best")
            else:
                ax.text(0.5, 0.5, "Curve unavailable",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{target} — {model_name}")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    combined_png = BASE / "HMF_Learning_Curves_Combined.png"
    combined_pdf = BASE / "HMF_Learning_Curves_Combined.pdf"
    fig.savefig(combined_png, dpi=300, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(curve_rows).to_csv(BASE / "learning_curve_values.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(BASE / "regression_model_comparison.csv", index=False)

    print("Finished.")
    print(f"Combined figure (PNG): {combined_png}")
    print(f"Combined figure (PDF): {combined_pdf}")
    print(f"Learning-curve values: {BASE / 'learning_curve_values.csv'}")
    print(f"Regression metrics: {BASE / 'regression_model_comparison.csv'}")
    print("Only one combined learning-curve figure is generated; no individual plots or new output folder.")
    print("Negative validation R² is retained and indicates performance worse than the test-fold mean baseline.")


if __name__ == "__main__":
    main()
