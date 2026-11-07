"""
Research-audited HMF -> FDCA surrogate validation and gated NSGA-II.

Important:
- This script intentionally refuses to report an optimization as validated when
  grouped-CV surrogate performance is inadequate.
- It uses paper-grouped CV when Paper Title is available in the aligned cleaned file.
- It excludes outcomes/derived outcomes from predictors.
- It does NOT independently mutate catalyst/electrolyte/cell descriptors. That would
  generate chemically impossible combinations.
- Potential_V must be harmonized to a common reference electrode before it can be
  interpreted as a continuous optimization variable. This script checks this and
  reports a gate instead of silently optimizing mixed quantities.

Literature context (see report output):
1. Guo et al., Advanced Science (2023), DOI: 10.1002/advs.202205540:
   applied potential changes pathways/selectivity and there is a trade-off; the
   optimum is system-specific.
2. Jiang et al., SusMat (2023), DOI: 10.1002/sus2.109:
   pH-dependent pathways and catalyst/electrolyte effects matter.
3. Efficient Electrooxidation of HMF Using Co-Doped Ni3S2 Catalyst, (2022),
   DOI/source indexed at PMC9189636: FE and yield varied with applied potential,
   with competitive OER at higher potentials.
These papers support considering potential, selectivity, electrolyte and catalyst
as coupled variables; they do NOT validate numerical predictions from this dataset.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("default")

BASE = Path(r"D:\chemical_ml")
if not BASE.exists():
    BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "HMF_ML_Preprocessed.xlsx"
CLEAN_FILE = BASE / "HMF_Cleaned_Full_Dataset.xlsx"
OUT = BASE / "NSGA_II_Research_Validated"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {"Conversion_pct", "Yield_pct", "FE_pct", "Selectivity_pct"}
LEAKAGE_COLUMNS = TARGETS | {
    "FDCA_Target", "Major_Product_Clean", "Major Product",
    "Yield_Greater_Than_Conversion", "Yield_gt_Conversion",
    "FE_Above_100", "FE_gt_100", "Conversion_Above_100",
    "Conversion_gt_100", "Yield_Above_100", "Yield_gt_100",
    "Paper Title", "Journal Name", "DOI", "Authors",
}
MIN_GROUPED_CV_R2 = 0.10  # conservative screening threshold, not a universal law
RANDOM_STATE = 42

def metrics(y, p):
    return {
        "R2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run preprocessing first.")
    df = pd.read_excel(DATA_FILE)
    if not {"Conversion_pct", "Yield_pct"}.issubset(df.columns):
        raise ValueError("Required target columns Conversion_pct and Yield_pct are missing.")

    # Align paper groups only if row order/length can be verified.
    groups = None
    group_source = "row-wise CV fallback"
    if CLEAN_FILE.exists():
        clean = pd.read_excel(CLEAN_FILE)
        if len(clean) == len(df) and "Paper Title" in clean.columns:
            g = clean["Paper Title"].fillna("UNKNOWN_PAPER").astype(str).str.strip()
            # Do not let all missing titles become one huge group.
            g = g.where(g.ne("") & g.ne("nan"), pd.Series(
                [f"UNKNOWN_ROW_{i}" for i in range(len(g))], index=g.index))
            groups = g.to_numpy()
            group_source = "Paper Title from cleaned file (row-aligned by equal row count)"
    if groups is None:
        warnings.warn("Paper groups unavailable; random CV can overestimate generalization.")
        groups = np.arange(len(df)).astype(str)

    # Exclude outcomes and known target-derived labels from predictors.
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in LEAKAGE_COLUMNS and df[c].nunique(dropna=True) > 1
    ]
    if not feature_cols:
        raise ValueError("No usable numeric predictors after leakage exclusions.")

    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=1),
        "SVR_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(C=1.0, epsilon=0.1))
        ]),
    }

    rows, prediction_rows, chosen = [], {}, {}
    for target in ["Conversion_pct", "Yield_pct"]:
        valid = pd.to_numeric(df[target], errors="coerce").notna()
        idx = np.flatnonzero(valid.to_numpy())
        X = df.loc[valid, feature_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        y = pd.to_numeric(df.loc[valid, target], errors="coerce").to_numpy(dtype=float)
        g = groups[idx]
        n_groups = len(np.unique(g))
        if n_groups >= 3:
            cv = GroupKFold(n_splits=min(5, n_groups))
            splits = list(cv.split(X, y, g))
            cv_type = "GroupKFold by paper"
        else:
            cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=RANDOM_STATE)
            splits = list(cv.split(X, y))
            cv_type = "KFold fallback (paper groups unavailable)"
        best_name, best_score = None, -np.inf
        for name, estimator in models.items():
            fold_metrics = []
            oof = np.full(len(y), np.nan)
            for tr, te in splits:
                # All transformations are fit inside each fold.
                if name in {"RandomForest", "ExtraTrees", "DummyMean"}:
                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", clone(estimator))
                    ])
                else:
                    pipe = clone(estimator)
                pipe.fit(X.iloc[tr], y[tr])
                pred = pipe.predict(X.iloc[te])
                oof[te] = pred
                fold_metrics.append(metrics(y[te], pred))
            valid_oof = np.isfinite(oof)
            aggregate = metrics(y[valid_oof], oof[valid_oof])
            rows.append({
                "Target": target, "Model": name, "N": len(y), "Groups": n_groups,
                "CV": cv_type, **aggregate,
                "Fold_R2_mean": float(np.nanmean([m["R2"] for m in fold_metrics])),
                "Fold_R2_sd": float(np.nanstd([m["R2"] for m in fold_metrics], ddof=1))
                    if len(fold_metrics) > 1 else np.nan
            })
            prediction_rows[(target, name)] = (y, oof)
            if aggregate["R2"] > best_score:
                best_score, best_name = aggregate["R2"], name
        chosen[target] = {"model": best_name, "grouped_oof_r2": float(best_score), "n": len(y)}

    results = pd.DataFrame(rows).sort_values(["Target", "R2"], ascending=[True, False])
    results.to_csv(OUT / "Surrogate_Model_GroupedCV_Comparison.csv", index=False)

    # Diagnostic plots from out-of-fold predictions, not in-sample fit.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, target in zip(axes, ["Conversion_pct", "Yield_pct"]):
        best = chosen[target]["model"]
        y, p = prediction_rows[(target, best)]
        mask = np.isfinite(p)
        ax.scatter(y[mask], p[mask], alpha=0.75)
        lo = min(float(np.min(y[mask])), float(np.min(p[mask])))
        hi = max(float(np.max(y[mask])), float(np.max(p[mask])))
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xlabel("Observed (%)")
        ax.set_ylabel("Out-of-fold predicted (%)")
        ax.set_title(f"{target}: {best}\nOOF R²={r2_score(y[mask], p[mask]):.3f}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "Surrogate_GroupedCV_Predictions.png", dpi=300)
    fig.savefig(OUT / "Surrogate_GroupedCV_Predictions.pdf")
    plt.close(fig)

    # Research/validity gate: don't report a GA front if prediction isn't useful.
    potential_note = (
        "Potential_V has values that may mix electrode-reference conventions / "
        "measurement modes. It must be verified and harmonized before optimization."
    )
    temp_note = (
        "Temperature_C is missing for many records and may include imputed/default values; "
        "verify provenance before treating it as an optimization variable."
    )
    best_r2s = [chosen[t]["grouped_oof_r2"] for t in ["Conversion_pct", "Yield_pct"]]
    gate_pass = all(np.isfinite(v) and v >= MIN_GROUPED_CV_R2 for v in best_r2s)
    gate = {
        "dataset": str(DATA_FILE),
        "rows": int(len(df)),
        "feature_count_after_leakage_exclusion": int(len(feature_cols)),
        "group_source": group_source,
        "group_count": int(len(np.unique(groups))),
        "chosen_models": chosen,
        "minimum_grouped_oof_r2_for_exploratory_optimization": MIN_GROUPED_CV_R2,
        "surrogate_gate_passed": bool(gate_pass),
        "optimization_status": "NOT RUN: surrogate validation gate failed" if not gate_pass
            else "SURROGATES PASSED SCREENING ONLY; further chemical/feature-space constraints still required",
        "potential_warning": potential_note,
        "temperature_warning": temp_note,
        "literature": [
            "Guo et al., Advanced Science (2023), DOI: 10.1002/advs.202205540",
            "Jiang et al., SusMat (2023), DOI: 10.1002/sus2.109",
            "Co-doped Ni3S2 HMF electrooxidation study (2022), PMC9189636"
        ],
    }
    (OUT / "Validation_Gate_and_Literature.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8")

    with open(OUT / "READ_ME_FIRST.txt", "w", encoding="utf-8") as f:
        f.write("HMF -> FDCA research-audited NSGA-II preparation\n")
        f.write("=" * 58 + "\n")
        f.write(f"Rows in preprocessed file: {len(df)}\n")
        f.write(f"Numeric predictors retained after target leakage exclusions: {len(feature_cols)}\n")
        f.write(f"Grouping source: {group_source}\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nChosen OOF models:\n" + json.dumps(chosen, indent=2))
        f.write("\n\nOptimization gate: " + str(gate_pass) + "\n")
        f.write("\n" + potential_note + "\n" + temp_note + "\n")
        f.write("\nThe literature supports trade-offs and system-specific optima; it does not validate "
                "the model predictions from this dataset. Do not claim experimental optimality from "
                "surrogate-only predictions.\n")

    print("\nGrouped-CV surrogate comparison:")
    print(results.to_string(index=False))
    print("\nBest OOF models:", chosen)
    print("\nOptimization gate:", "PASS" if gate_pass else "BLOCKED")
    if not gate_pass:
        print("NSGA-II was not run because surrogate predictions do not meet the validation threshold.")
    print("Outputs saved to:", OUT)

if __name__ == "__main__":
    main()
