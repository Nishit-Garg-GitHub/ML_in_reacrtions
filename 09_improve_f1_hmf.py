# ============================================================
# HMF -> FDCA: DEFENSIBLE F1 MODEL COMPARISON
# Python 3.12
#
# Goal:
#   Compare models for predicting performance clusters derived
#   from Conversion and FDCA Yield, while avoiding paper leakage.
#
# Important:
#   - Conversion/Yield define labels, NEVER predictors.
#   - All records from the same paper stay in one CV fold.
#   - Imputation, scaling and oversampling occur inside pipelines.
#   - Macro-F1 is the primary metric; weighted-F1 is secondary.
#   - Results are estimates, not guaranteed improvements.
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings('ignore')

# -------- Paths: edit only if your project folder differs --------
BASE = Path(r'D:\chemical_ml')
DATA_FILE = BASE / 'HMF_ML_Preprocessed.xlsx'
CLEAN_FILE = BASE / 'HMF_Cleaned_Full_Dataset.xlsx'
OUTPUT_DIR = BASE / 'F1_Improvement_Results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_VALUES = [3, 4, 5, 6]
MAX_OUTER_SPLITS = 4

# Never use outcome, outcome-derived flags, or paper identifiers as predictors.
TARGET_OR_LEAKAGE_COLUMNS = {
    'Conversion_pct', 'Yield_pct', 'FE_pct', 'Selectivity_pct',
    'FDCA_Target', 'Major Product', 'Yield_gt_Conversion',
    'Yield_Greater_Than_Conversion', 'Conversion_gt_100',
    'Conversion_Above_100', 'Yield_gt_100', 'Yield_Above_100',
    'FE_gt_100', 'FE_Above_100', 'Potential_vs_RHE',
    'Paper Title', 'DOI', 'Author', 'Authors', 'Reference',
    'Citation', 'Publication', 'Year'
}


def load_aligned_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Cannot find {DATA_FILE}')
    df = pd.read_excel(DATA_FILE)
    required = {'Conversion_pct', 'Yield_pct'}
    if not required.issubset(df.columns):
        raise ValueError(f'Required columns missing: {required - set(df.columns)}')

    mask = df['Conversion_pct'].notna() & df['Yield_pct'].notna()
    original_positions = np.flatnonzero(mask.to_numpy())
    data = df.loc[mask].reset_index(drop=True)

    # Prefer publication grouping from the cleaned full dataset.
    groups = None
    if CLEAN_FILE.exists():
        clean = pd.read_excel(CLEAN_FILE)
        if len(clean) == len(df) and 'Paper Title' in clean.columns:
            groups = clean.iloc[original_positions]['Paper Title'].reset_index(drop=True)
        else:
            for col in ['Paper Title', 'Paper', 'Reference', 'DOI', 'Citation']:
                if col in data.columns:
                    groups = data[col]
                    break
    else:
        for col in ['Paper Title', 'Paper', 'Reference', 'DOI', 'Citation']:
            if col in data.columns:
                groups = data[col]
                break

    if groups is None:
        raise ValueError(
            'Could not identify a paper/publication grouping column. '
            'Add a Paper Title column to the cleaned dataset; grouped validation '
            'is needed to test generalization across publications.'
        )
    groups = groups.fillna('UNKNOWN_PAPER').astype(str).to_numpy()

    # Numeric features only: current pipeline uses numeric/one-hot encoded features.
    feature_cols = []
    for c in data.select_dtypes(include=[np.number]).columns:
        if c in TARGET_OR_LEAKAGE_COLUMNS:
            continue
        if data[c].notna().sum() == 0 or data[c].nunique(dropna=True) <= 1:
            continue
        feature_cols.append(c)

    X = data[feature_cols].apply(pd.to_numeric, errors='coerce')
    # Remove near-constant binary columns; these cannot generalize reliably at this N.
    for c in list(X.columns):
        vals = X[c].dropna()
        if len(vals) and vals.nunique() <= 2 and set(vals.unique()).issubset({0, 1}):
            p = vals.mean()
            if p < 0.02 or p > 0.98:
                X.drop(columns=c, inplace=True)
    return data, X, groups


def build_labels(data, k):
    """KMeans classes in standardized Conversion/Yield outcome space."""
    outcomes = data[['Conversion_pct', 'Yield_pct']].astype(float).to_numpy()
    outcomes = np.clip(outcomes, 0, 100)
    # Standardize the two percentage outcomes before distance-based clustering.
    scaler = StandardScaler()
    z = scaler.fit_transform(outcomes)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
    labels = km.fit_predict(z)
    # Relabel clusters by ascending average combined performance for consistency.
    score = outcomes.mean(axis=1)
    order = sorted(np.unique(labels), key=lambda cl: float(score[labels == cl].mean()))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[v] for v in labels], dtype=int)


def make_pipeline(model_name, sampler_name, model):
    steps = [('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())]
    if sampler_name == 'ROS':
        steps.append(('sampler', RandomOverSampler(random_state=RANDOM_STATE)))
    elif sampler_name == 'SMOTE':
        steps.append(('sampler', SMOTE(random_state=RANDOM_STATE, k_neighbors=1)))
    steps.append(('model', model))
    # All samplers are inside imblearn Pipeline and are fit on each training fold only.
    return ImbPipeline(steps)


def get_model_specs():
    return {
        'Dummy': (DummyClassifier(strategy='most_frequent'), {}),
        'DecisionTree': (DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'), {
            'model__max_depth': [2, 3, 5, None],
            'model__min_samples_leaf': [2, 4, 6]
        }),
        'SVM': (SVC(class_weight='balanced', probability=False, random_state=RANDOM_STATE), {
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 0.01, 0.1],
            'model__kernel': ['rbf']
        }),
        'RandomForest': (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced_subsample'), {
            'model__n_estimators': [200, 400],
            'model__max_depth': [3, 6, None],
            'model__min_samples_leaf': [2, 4, 6],
            'model__max_features': ['sqrt', 0.7]
        }),
        'ExtraTrees': (ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'), {
            'model__n_estimators': [200, 400],
            'model__max_depth': [3, 6, None],
            'model__min_samples_leaf': [2, 4, 6],
            'model__max_features': ['sqrt', 0.7]
        })
    }


def main():
    data, X, groups = load_aligned_data()
    print(f'Rows with Conversion and Yield: {len(data)}')
    print(f'Predictor count: {X.shape[1]}')
    print('Predictors:', ', '.join(X.columns))
    print(f'Unique paper groups: {len(np.unique(groups))}')
    if X.shape[1] == 0:
        raise ValueError('No usable numeric predictor columns found.')

    all_results = []
    all_oof = []
    model_specs = get_model_specs()

    for k in K_VALUES:
        y = build_labels(data, k)
        counts = pd.Series(y).value_counts().sort_index()
        print(f'\n===== K={k}; class counts={counts.to_dict()} =====')
        if counts.min() < 2:
            print(f'SKIP K={k}: KMeans produced a singleton class; F1 would be unstable and SMOTE invalid.')
            continue

        # Use the largest feasible number of folds given rare classes and publication groups.
        n_splits = min(MAX_OUTER_SPLITS, int(counts.min()), len(np.unique(groups)))
        if n_splits < 2:
            print(f'SKIP K={k}: insufficient class/group support for grouped CV.')
            continue
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        oof_by_config = {}

        for model_name, (base_model, grid) in model_specs.items():
            samplers = ['NONE'] if model_name == 'Dummy' else ['NONE', 'ROS', 'SMOTE']
            for sampler_name in samplers:
                print(f'  Evaluating {model_name} | {sampler_name}')
                y_true_all, y_pred_all = [], []
                fold_macro, fold_weighted = [], []
                failed = 0

                for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
                    Xtr, Xte = X.iloc[tr], X.iloc[te]
                    ytr, yte = y[tr], y[te]
                    gtr = groups[tr]
                    # Need enough examples of every class for inner grouped CV.
                    train_counts = pd.Series(ytr).value_counts()
                    inner_n = min(3, int(train_counts.min()), len(np.unique(gtr)))
                    try:
                        pipe = make_pipeline(model_name, sampler_name, base_model)
                        if grid and inner_n >= 2:
                            inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True,
                                                        random_state=RANDOM_STATE + fold)
                            search = GridSearchCV(
                                pipe, grid, scoring='f1_macro', cv=inner,
                                n_jobs=-1, refit=True, error_score=np.nan
                            )
                            search.fit(Xtr, ytr, groups=gtr)
                            fitted = search.best_estimator_
                        else:
                            fitted = pipe.fit(Xtr, ytr)
                        pred = fitted.predict(Xte)
                        y_true_all.extend(yte.tolist())
                        y_pred_all.extend(pred.tolist())
                        fold_macro.append(f1_score(yte, pred, average='macro', zero_division=0))
                        fold_weighted.append(f1_score(yte, pred, average='weighted', zero_division=0))
                    except Exception as e:
                        failed += 1
                        print(f'    fold {fold} skipped: {type(e).__name__}: {e}')

                if not y_true_all or failed:
                    print(f'    Not reported: {failed} fold(s) failed; do not compare incomplete CV results.')
                    continue

                macro = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                weighted = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
                key = (k, model_name, sampler_name)
                oof_by_config[key] = (np.asarray(y_true_all), np.asarray(y_pred_all))
                result = {
                    'K': k, 'Model': model_name, 'Sampling': sampler_name,
                    'Macro_F1_OOF': macro, 'Weighted_F1_OOF': weighted,
                    'Accuracy_OOF': accuracy_score(y_true_all, y_pred_all),
                    'Balanced_Accuracy_OOF': balanced_accuracy_score(y_true_all, y_pred_all),
                    'Precision_Macro_OOF': precision_score(y_true_all, y_pred_all, average='macro', zero_division=0),
                    'Recall_Macro_OOF': recall_score(y_true_all, y_pred_all, average='macro', zero_division=0),
                    'Fold_Macro_F1_Mean': float(np.mean(fold_macro)),
                    'Fold_Macro_F1_SD': float(np.std(fold_macro, ddof=1)) if len(fold_macro) > 1 else 0.0,
                    'N': len(y_true_all), 'Features': X.shape[1], 'Folds': n_splits
                }
                all_results.append(result)
                print(f"    Macro-F1={macro:.3f}; weighted-F1={weighted:.3f}; balanced accuracy={result['Balanced_Accuracy_OOF']:.3f}")

        # Save out-of-fold predictions and per-class report for K.
        for key, (yt, yp) in oof_by_config.items():
            _, model_name, sampler_name = key
            pred_df = pd.DataFrame({'K': k, 'Model': model_name, 'Sampling': sampler_name,
                                    'True_Class': yt, 'Predicted_Class': yp})
            all_oof.append(pred_df)

    if not all_results:
        raise RuntimeError('No complete grouped-CV results. Inspect class counts and paper-group counts.')

    results = pd.DataFrame(all_results).sort_values(['K', 'Macro_F1_OOF'], ascending=[True, False])
    results.to_csv(OUTPUT_DIR / 'F1_model_comparison.csv', index=False)
    if all_oof:
        pd.concat(all_oof, ignore_index=True).to_csv(OUTPUT_DIR / 'F1_oof_predictions.csv', index=False)

    # Save the best configuration for each K, based on macro-F1.
    best = results.loc[results.groupby('K')['Macro_F1_OOF'].idxmax()].sort_values('K')
    best.to_csv(OUTPUT_DIR / 'Best_model_per_K_macroF1.csv', index=False)

    print('\n===== BEST CONFIGURATION BY K (macro-F1) =====')
    print(best[['K', 'Model', 'Sampling', 'Macro_F1_OOF', 'Weighted_F1_OOF', 'Balanced_Accuracy_OOF', 'Fold_Macro_F1_SD']].to_string(index=False))
    print(f'\nSaved results to: {OUTPUT_DIR}')
    print('Primary selection metric: Macro-F1. Do not choose a model from training score.')

if __name__ == '__main__':
    main()
