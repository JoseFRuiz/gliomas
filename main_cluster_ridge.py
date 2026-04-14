# =============================================================================
# Cluster-Guided Ridge Regression Pipeline
# =============================================================================
# Pipeline overview:
#   1. Load & preprocess data  (log-transform survival, feature selection)
#   2. Cluster samples with HDBSCAN on reduced gene expression space
#   3. Evaluate per-cluster CV performance (Ridge R²)
#   4. Greedily filter clusters to maximise:  R²  -  λ × fraction_removed
#   5. Re-evaluate Ridge CV on the retained subset
#   6. Train & evaluate two classifiers (Logistic Regression + Random Forest)
#      to separate retained vs. discarded samples
#   7. Report feature importance / gene weights from both classifiers
#   8. Save all outputs (CSVs + plots)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

from utils import load_data, cross_validate_regression

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================

GENE_TPM_PATH        = os.path.join('data', 'TCGAGliomas_RNAm_Filtrado_QC_verif.csv')
LOG_TRANSFORM        = True          # log(1+x) on survival days

# Feature selection
N_FEATURES           = 1000          # top features by F-regression for the Ridge model
N_PCA_COMPONENTS     = 30           # PCA dims fed to HDBSCAN (reduces noise)

# HDBSCAN
HDBSCAN_MIN_CLUSTER  = 5            # min_cluster_size  (tune: larger → fewer, bigger clusters)
HDBSCAN_MIN_SAMPLES  = 3            # min_samples       (tune: larger → more noise points)

# Cluster filtering
LAMBDA_PENALTY       = 0.5          # λ in:  score = R²_cv  -  λ × fraction_removed
#   • λ=0   → pure R² maximisation (may discard many samples)
#   • λ=1   → equal weight on keeping samples (conservative)
#   • λ=0.5 → balanced default
MAX_FRACTION_REMOVE  = 0.50         # hard cap: never discard more than this fraction

# Ridge
RIDGE_ALPHA          = 100.0

# Cross-validation
N_CV_FOLDS           = 5

# Classifier
CLF_MAX_ITER         = 2000
RF_N_ESTIMATORS      = 300
TOP_N_FEATURES_SHOW  = 20           # how many genes to plot in importance charts

# Output
OUTPUT_DIR           = 'output'
RANDOM_STATE         = 42

# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

def select_features(X, y, n_features):
    """Return top-n feature names by F-regression score."""
    X_vals = X.values if isinstance(X, pd.DataFrame) else X
    y_vals = y.values if isinstance(y, pd.Series) else y
    f_scores, _ = f_regression(X_vals, y_vals)
    top_idx = np.argsort(f_scores)[-n_features:][::-1]
    if isinstance(X, pd.DataFrame):
        return X.columns[top_idx].tolist()
    return top_idx.tolist()


def cv_r2(X_arr, y_arr, alpha=RIDGE_ALPHA, n_splits=N_CV_FOLDS, rs=RANDOM_STATE):
    """5-fold CV R² for Ridge.  Returns NaN if too few samples."""
    n = len(y_arr)
    if n < 2 * n_splits:
        return np.nan
    splits = min(n_splits, n // 2)
    cv = KFold(n_splits=splits, shuffle=True, random_state=rs)
    scores = []
    for tr, va in cv.split(X_arr):
        if len(va) < 2:
            continue
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_arr[tr], y_arr[tr])
        scores.append(r2_score(y_arr[va], mdl.predict(X_arr[va])))
    return float(np.mean(scores)) if scores else np.nan


def penalised_score(r2_val, n_kept, n_total, lam=LAMBDA_PENALTY):
    """score = R²  -  λ × fraction_removed   (NaN R² → -inf)."""
    if np.isnan(r2_val):
        return -np.inf
    frac_removed = 1.0 - n_kept / n_total
    return r2_val - lam * frac_removed


def safe_auc(y_true, y_prob):
    """ROC-AUC robust to single-class folds."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


# =============================================================================
# ── STEP 0 : Load & prepare data ──────────────────────────────────────────────
# =============================================================================
print("=" * 65)
print("CLUSTER-GUIDED RIDGE PIPELINE")
print("=" * 65)

X, y = load_data(gene_tpm_path=GENE_TPM_PATH)

if LOG_TRANSFORM:
    y = pd.Series(np.log1p(y.values), index=y.index, name=y.name)
    target_label = 'log(1 + Survival Days)'
else:
    target_label = 'Survival Days'

n_total = X.shape[0]
print(f"\n[0] Data loaded  →  {n_total} samples × {X.shape[1]} genes")
print(f"    Target: {target_label}  range [{y.min():.2f}, {y.max():.2f}]")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# ── STEP 1 : Feature selection  ───────────────────────────────────────────────
# =============================================================================
print(f"\n[1] Selecting top {N_FEATURES} features by F-regression …")
selected_features = select_features(X, y, N_FEATURES)
X_sel = X[selected_features]          # (n_samples × N_FEATURES) DataFrame

# =============================================================================
# ── STEP 2 : HDBSCAN clustering on PCA-reduced space ─────────────────────────
# =============================================================================
print(f"\n[2] Clustering with HDBSCAN  (PCA={N_PCA_COMPONENTS} dims, "
      f"min_cluster_size={HDBSCAN_MIN_CLUSTER}) …")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel.values)

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
              min_samples=HDBSCAN_MIN_SAMPLES)
cluster_labels = hdb.fit_predict(X_pca)   # -1 = noise / unclustered

unique_clusters = sorted(set(cluster_labels))
n_clusters      = sum(1 for c in unique_clusters if c >= 0)
n_noise         = np.sum(cluster_labels == -1)

print(f"    Clusters found : {n_clusters}  (+ {n_noise} noise points labelled -1)")
print(f"    Cluster sizes  : { {c: int(np.sum(cluster_labels==c)) for c in unique_clusters} }")

# Build a Series indexed like y
cluster_series = pd.Series(cluster_labels, index=X.index, name='cluster')

# =============================================================================
# ── STEP 3 : Per-cluster CV performance ───────────────────────────────────────
# =============================================================================
print(f"\n[3] Evaluating per-cluster Ridge CV R² …")

X_arr = X_sel.values.astype(float)
y_arr = y.values.astype(float)

cluster_stats = {}   # cluster_id → {n, r2, score}
for cid in unique_clusters:
    mask = cluster_labels == cid
    n_c  = mask.sum()
    r2_c = cv_r2(X_arr[mask], y_arr[mask])
    cluster_stats[cid] = {'n': int(n_c), 'r2': r2_c}
    tag = "NOISE" if cid == -1 else f"Cluster {cid}"
    r2_str = f"{r2_c:.4f}" if not np.isnan(r2_c) else "N/A (too small)"
    print(f"    {tag:12s}  n={n_c:4d}   R²={r2_str}")

# =============================================================================
# ── STEP 4 : Greedy cluster filtering  (penalised score) ─────────────────────
# =============================================================================
print(f"\n[4] Greedy cluster filtering  (λ={LAMBDA_PENALTY}, "
      f"max_remove={MAX_FRACTION_REMOVE*100:.0f}%) …")

# Baseline: ALL samples (including noise points)
baseline_r2    = cv_r2(X_arr, y_arr)
baseline_score = penalised_score(baseline_r2, n_total, n_total)
print(f"    Baseline  →  R²={baseline_r2:.4f}   penalised={baseline_score:.4f}  "
      f"(n={n_total})")

# =============================================================================
# ── STEP 3b : Discard-one-cluster performance (leave-one-cluster-out) ─────
# =============================================================================
print(f"\n[3b] Discard-one-cluster performance (leave-one-cluster-out) …")
leave_one_out_rows = []
for cid in unique_clusters:
    keep_mask = cluster_labels != cid
    n_kept = int(keep_mask.sum())
    n_disc = int(n_total - n_kept)
    r2_without = cv_r2(X_arr[keep_mask], y_arr[keep_mask])
    score_without = penalised_score(r2_without, n_kept, n_total)

    tag = "NOISE (-1)" if cid == -1 else f"Cluster {cid}"
    cluster_size = int(np.sum(cluster_labels == cid))
    r2_str = f"{r2_without:.4f}" if not np.isnan(r2_without) else "N/A (too small)"
    print(f"    Discard {tag:12s}  removed_n={cluster_size:4d}  kept={n_kept:4d}  "
          f"R²={r2_str}  penalised={score_without:.4f}")

    leave_one_out_rows.append({
        'removed_cluster': int(cid),
        'cluster_tag': tag,
        'removed_n': cluster_size,
        'n_kept': n_kept,
        'n_discarded': n_disc,
        'r2_without_cluster': float(r2_without) if not np.isnan(r2_without) else np.nan,
        'penalised_score_without_cluster': float(score_without),
    })

# Extra experiment: keep ONLY the noise cluster (-1), discard everything else
noise_only_mask = (cluster_labels == -1)
n_noise = int(noise_only_mask.sum())
if n_noise > 0:
    r2_noise_only = cv_r2(X_arr[noise_only_mask], y_arr[noise_only_mask])
    score_noise_only = penalised_score(r2_noise_only, n_noise, n_total)
    print(f"    Discard ALL non-NOISE  removed_n={n_total - n_noise:4d}  kept={n_noise:4d}  "
          f"R²={r2_noise_only:.4f}  penalised={score_noise_only:.4f}")
    leave_one_out_rows.append({
        'removed_cluster': 'KEEP_ONLY_-1',
        'cluster_tag': 'Retain ONLY NOISE (-1)',
        'removed_n': int(n_total - n_noise),
        'n_kept': n_noise,
        'n_discarded': int(n_total - n_noise),
        'r2_without_cluster': float(r2_noise_only) if not np.isnan(r2_noise_only) else np.nan,
        'penalised_score_without_cluster': float(score_noise_only),
    })

f_loo = os.path.join(OUTPUT_DIR, 'cluster_leave_one_out_ridge_performance.csv')
pd.DataFrame(leave_one_out_rows).to_csv(f_loo, index=False)
print(f"    Saved: {f_loo}")

# We treat noise cluster (-1) like any other cluster for filtering purposes.
# The algorithm greedily removes the cluster whose removal most improves the
# penalised score, stopping when no removal helps.

active_clusters = set(unique_clusters)          # clusters currently kept
removal_log     = []                             # track decisions

def mask_from_active(active):
    return np.isin(cluster_labels, list(active))

current_mask  = mask_from_active(active_clusters)
current_r2    = baseline_r2
current_score = baseline_score

for step in range(len(unique_clusters)):
    best_delta    = 0.0
    best_cluster  = None
    best_r2       = current_r2
    best_score    = current_score

    for cid in list(active_clusters):
        # Only treat the noise cluster (-1) as a separate experiment (Step 3b),
        # not as an option inside the greedy removal loop.
        if cid == -1:
            continue
        candidate_active = active_clusters - {cid}
        cand_mask  = mask_from_active(candidate_active)
        n_kept     = cand_mask.sum()
        frac_rem   = 1.0 - n_kept / n_total

        # Hard cap on fraction removed
        if frac_rem > MAX_FRACTION_REMOVE:
            continue
        # Need enough samples for CV
        if n_kept < 2 * N_CV_FOLDS:
            continue

        r2_cand    = cv_r2(X_arr[cand_mask], y_arr[cand_mask])
        sc_cand    = penalised_score(r2_cand, n_kept, n_total)
        delta      = sc_cand - current_score

        if delta > best_delta:
            best_delta   = delta
            best_cluster = cid
            best_r2      = r2_cand
            best_score   = sc_cand

    if best_cluster is None:
        print(f"    Step {step+1}: No further improvement — stopping.")
        break

    # Accept removal
    active_clusters.remove(best_cluster)
    current_mask  = mask_from_active(active_clusters)
    current_r2    = best_r2
    current_score = best_score
    n_kept_now    = current_mask.sum()
    tag = "NOISE" if best_cluster == -1 else f"Cluster {best_cluster}"
    removal_log.append({
        'step': step + 1, 'removed_cluster': best_cluster,
        'cluster_tag': tag,
        'cluster_size': cluster_stats[best_cluster]['n'],
        'r2_after': current_r2, 'penalised_score': current_score,
        'n_kept': n_kept_now
    })
    print(f"    Step {step+1}: Remove {tag:12s}  "
          f"(n={cluster_stats[best_cluster]['n']:3d})  →  "
          f"R²={current_r2:.4f}  penalised={current_score:.4f}  "
          f"kept={n_kept_now}/{n_total}")

# Final retained / discarded index lists
retained_mask    = current_mask
discarded_mask   = ~retained_mask
retained_idx     = X.index[retained_mask]
discarded_idx    = X.index[discarded_mask]
n_retained       = retained_mask.sum()
n_discarded      = discarded_mask.sum()

print(f"\n    ── Filtering result ──")
print(f"    Retained  : {n_retained} samples ({100*n_retained/n_total:.1f}%)")
print(f"    Discarded : {n_discarded} samples ({100*n_discarded/n_total:.1f}%)")
print(f"    Final CV R²          : {current_r2:.4f}   (baseline: {baseline_r2:.4f})")
print(f"    Δ R²                 : {current_r2 - baseline_r2:+.4f}")

# =============================================================================
# ── STEP 5 : Full CV evaluation on retained subset ────────────────────────────
# =============================================================================
print(f"\n[5] Final cross-validation on retained subset …")

X_ret  = X_sel.loc[retained_idx]
y_ret  = y.loc[retained_idx]

results_retained = cross_validate_regression(
    X_ret, y_ret, model=Ridge(alpha=RIDGE_ALPHA),
    model_name='Ridge (Retained)', cv=N_CV_FOLDS)

results_full = cross_validate_regression(
    X_sel, y, model=Ridge(alpha=RIDGE_ALPHA),
    model_name='Ridge (Full)', cv=N_CV_FOLDS)

print(f"    Full dataset     →  R²={results_full['mean_score']:.4f} ± "
      f"{results_full['std_score']:.4f}   corr={results_full['correlation']:.4f}")
print(f"    Retained subset  →  R²={results_retained['mean_score']:.4f} ± "
      f"{results_retained['std_score']:.4f}   corr={results_retained['correlation']:.4f}")

# =============================================================================
# ── STEP 6 : Classifier — retained vs. discarded ─────────────────────────────
# =============================================================================
print(f"\n[6] Training classifiers (retained=1 vs. discarded=0) …")

y_binary = pd.Series(
    (X.index.isin(retained_idx)).astype(int),
    index=X.index, name='retained')

X_clf_arr   = X_arr          # full dataset, selected features
y_clf_arr   = y_binary.values

n_pos = y_clf_arr.sum()
n_neg = len(y_clf_arr) - n_pos
print(f"    Class balance  retained={n_pos}  discarded={n_neg}")

n_splits_clf = min(N_CV_FOLDS, n_pos, n_neg)

# Metadata / outputs for classifier evaluation
n_features_used_for_classifier_cv = int(X_clf_arr.shape[1])
lr_train_test_rows = []
rf_train_test_rows = []
lr_cv_summary = {}
rf_cv_summary = {}

# Extra scenario: classifier if we KEEP ONLY the noise cluster (-1)
noise_lr_train_test_rows = []
noise_rf_train_test_rows = []
noise_lr_cv_summary = {}
noise_rf_cv_summary = {}

# ── 6a. Logistic Regression ──────────────────────────────────────────────────
print("    [LR]  Logistic Regression …")
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=CLF_MAX_ITER, class_weight='balanced',
                                  random_state=RANDOM_STATE, C=1.0))
])

if n_splits_clf >= 2:
    cv_clf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True,
                              random_state=RANDOM_STATE)
    acc_lr  = cross_val_score(pipe_lr, X_clf_arr, y_clf_arr,
                               cv=cv_clf, scoring='accuracy', n_jobs=-1)
    pred_lr = cross_val_predict(pipe_lr, X_clf_arr, y_clf_arr,
                                 cv=cv_clf, method='predict_proba',
                                 n_jobs=-1)[:, 1]
    auc_lr  = [safe_auc(y_clf_arr[list(va)],
                         pred_lr[list(va)])
               for _, va in cv_clf.split(X_clf_arr, y_clf_arr)]
    auc_lr  = [v for v in auc_lr if not np.isnan(v)]
    print(f"    [LR]  Accuracy={np.mean(acc_lr):.4f}±{np.std(acc_lr):.4f}  "
          f"ROC-AUC={np.mean(auc_lr):.4f}±{np.std(auc_lr):.4f}")

    # Per-fold train vs test performance (useful to spot overfitting)
    lr_train_accs = []
    lr_test_accs = []
    lr_train_aucs = []
    lr_test_aucs = []
    fold_idx = 0
    for tr_idx, va_idx in cv_clf.split(X_clf_arr, y_clf_arr):
        pipe_lr_fold = clone(pipe_lr)
        pipe_lr_fold.fit(X_clf_arr[tr_idx], y_clf_arr[tr_idx])

        # Train metrics
        y_pred_train = pipe_lr_fold.predict(X_clf_arr[tr_idx])
        train_acc = accuracy_score(y_clf_arr[tr_idx], y_pred_train)
        lr_train_accs.append(train_acc)

        train_auc = np.nan
        try:
            y_prob_train = pipe_lr_fold.predict_proba(X_clf_arr[tr_idx])[:, 1]
            train_auc = safe_auc(y_clf_arr[tr_idx], y_prob_train)
        except Exception:
            pass
        lr_train_aucs.append(train_auc)

        # Test/validation metrics
        y_pred_test = pipe_lr_fold.predict(X_clf_arr[va_idx])
        test_acc = accuracy_score(y_clf_arr[va_idx], y_pred_test)
        lr_test_accs.append(test_acc)

        test_auc = np.nan
        try:
            y_prob_test = pipe_lr_fold.predict_proba(X_clf_arr[va_idx])[:, 1]
            test_auc = safe_auc(y_clf_arr[va_idx], y_prob_test)
        except Exception:
            pass
        lr_test_aucs.append(test_auc)

        lr_train_test_rows.append({
            'classifier': 'LogisticRegression',
            'fold': fold_idx,
            'n_train': int(len(tr_idx)),
            'n_test': int(len(va_idx)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_roc_auc': float(train_auc) if not np.isnan(train_auc) else np.nan,
            'test_roc_auc': float(test_auc) if not np.isnan(test_auc) else np.nan,
        })
        fold_idx += 1

    lr_cv_summary = {
        'classifier': 'LogisticRegression',
        'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
        'n_splits': int(n_splits_clf),
        'train_accuracy_mean': float(np.mean(lr_train_accs)) if lr_train_accs else np.nan,
        'train_accuracy_std': float(np.std(lr_train_accs)) if lr_train_accs else np.nan,
        'test_accuracy_mean': float(np.mean(lr_test_accs)) if lr_test_accs else np.nan,
        'test_accuracy_std': float(np.std(lr_test_accs)) if lr_test_accs else np.nan,
        'train_roc_auc_mean': float(np.nanmean(lr_train_aucs)) if lr_train_aucs else np.nan,
        'train_roc_auc_std': float(np.nanstd(lr_train_aucs)) if lr_train_aucs else np.nan,
        'test_roc_auc_mean': float(np.nanmean(lr_test_aucs)) if lr_test_aucs else np.nan,
        'test_roc_auc_std': float(np.nanstd(lr_test_aucs)) if lr_test_aucs else np.nan,
    }
else:
    print("    [LR]  Skipped CV (too few samples per class)")
    acc_lr = auc_lr = []

# Fit on full data for coefficients
pipe_lr.fit(X_clf_arr, y_clf_arr)
coef_lr = pipe_lr.named_steps['clf'].coef_.ravel()
lr_importance = pd.DataFrame({
    'Feature':         selected_features,
    'LR_Coefficient':  coef_lr,
    'LR_Abs':          np.abs(coef_lr)
}).sort_values('LR_Abs', ascending=False).reset_index(drop=True)

# ── 6b. Random Forest ────────────────────────────────────────────────────────
print("    [RF]  Random Forest …")
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                      class_weight='balanced',
                                      random_state=RANDOM_STATE,
                                      n_jobs=-1))
])

if n_splits_clf >= 2:
    acc_rf  = cross_val_score(pipe_rf, X_clf_arr, y_clf_arr,
                               cv=cv_clf, scoring='accuracy', n_jobs=-1)
    pred_rf = cross_val_predict(pipe_rf, X_clf_arr, y_clf_arr,
                                 cv=cv_clf, method='predict_proba',
                                 n_jobs=-1)[:, 1]
    auc_rf  = [safe_auc(y_clf_arr[list(va)],
                         pred_rf[list(va)])
               for _, va in cv_clf.split(X_clf_arr, y_clf_arr)]
    auc_rf  = [v for v in auc_rf if not np.isnan(v)]
    print(f"    [RF]  Accuracy={np.mean(acc_rf):.4f}±{np.std(acc_rf):.4f}  "
          f"ROC-AUC={np.mean(auc_rf):.4f}±{np.std(auc_rf):.4f}")

    # Per-fold train vs test performance (useful to spot overfitting)
    rf_train_accs = []
    rf_test_accs = []
    rf_train_aucs = []
    rf_test_aucs = []
    fold_idx = 0
    for tr_idx, va_idx in cv_clf.split(X_clf_arr, y_clf_arr):
        pipe_rf_fold = clone(pipe_rf)
        pipe_rf_fold.fit(X_clf_arr[tr_idx], y_clf_arr[tr_idx])

        # Train metrics
        y_pred_train = pipe_rf_fold.predict(X_clf_arr[tr_idx])
        train_acc = accuracy_score(y_clf_arr[tr_idx], y_pred_train)
        rf_train_accs.append(train_acc)

        train_auc = np.nan
        try:
            y_prob_train = pipe_rf_fold.predict_proba(X_clf_arr[tr_idx])[:, 1]
            train_auc = safe_auc(y_clf_arr[tr_idx], y_prob_train)
        except Exception:
            pass
        rf_train_aucs.append(train_auc)

        # Test/validation metrics
        y_pred_test = pipe_rf_fold.predict(X_clf_arr[va_idx])
        test_acc = accuracy_score(y_clf_arr[va_idx], y_pred_test)
        rf_test_accs.append(test_acc)

        test_auc = np.nan
        try:
            y_prob_test = pipe_rf_fold.predict_proba(X_clf_arr[va_idx])[:, 1]
            test_auc = safe_auc(y_clf_arr[va_idx], y_prob_test)
        except Exception:
            pass
        rf_test_aucs.append(test_auc)

        rf_train_test_rows.append({
            'classifier': 'RandomForest',
            'fold': fold_idx,
            'n_train': int(len(tr_idx)),
            'n_test': int(len(va_idx)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_roc_auc': float(train_auc) if not np.isnan(train_auc) else np.nan,
            'test_roc_auc': float(test_auc) if not np.isnan(test_auc) else np.nan,
        })
        fold_idx += 1

    rf_cv_summary = {
        'classifier': 'RandomForest',
        'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
        'n_splits': int(n_splits_clf),
        'train_accuracy_mean': float(np.mean(rf_train_accs)) if rf_train_accs else np.nan,
        'train_accuracy_std': float(np.std(rf_train_accs)) if rf_train_accs else np.nan,
        'test_accuracy_mean': float(np.mean(rf_test_accs)) if rf_test_accs else np.nan,
        'test_accuracy_std': float(np.std(rf_test_accs)) if rf_test_accs else np.nan,
        'train_roc_auc_mean': float(np.nanmean(rf_train_aucs)) if rf_train_aucs else np.nan,
        'train_roc_auc_std': float(np.nanstd(rf_train_aucs)) if rf_train_aucs else np.nan,
        'test_roc_auc_mean': float(np.nanmean(rf_test_aucs)) if rf_test_aucs else np.nan,
        'test_roc_auc_std': float(np.nanstd(rf_test_aucs)) if rf_test_aucs else np.nan,
    }
else:
    print("    [RF]  Skipped CV (too few samples per class)")
    acc_rf = auc_rf = []

# =============================================================================
# Extra scenario: KEEP ONLY noise cluster (-1) for classifier
#   retained=1  iff cluster_labels == -1
#   discarded=0 iff cluster_labels != -1
# =============================================================================
noise_only_mask = (cluster_labels == -1)
n_pos_noise = int(np.sum(noise_only_mask))
n_neg_noise = int(len(noise_only_mask) - n_pos_noise)
print(f"    [NOISE-ONLY] Class balance  retained={n_pos_noise}  discarded={n_neg_noise}")
n_splits_clf_noise = min(N_CV_FOLDS, n_pos_noise, n_neg_noise)

if n_splits_clf_noise >= 2:
    cv_clf_noise = StratifiedKFold(n_splits=n_splits_clf_noise, shuffle=True,
                                     random_state=RANDOM_STATE)

    # --- Logistic Regression ---
    print("    [NOISE-ONLY][LR] Logistic Regression …")
    y_noise = noise_only_mask.astype(int)

    pipe_lr_noise = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=CLF_MAX_ITER, class_weight='balanced',
                                    random_state=RANDOM_STATE, C=1.0))
    ])
    lr_train_accs = []
    lr_test_accs = []
    lr_train_aucs = []
    lr_test_aucs = []
    fold_idx = 0
    for tr_idx, va_idx in cv_clf_noise.split(X_clf_arr, y_noise):
        pipe_lr_fold = clone(pipe_lr_noise)
        pipe_lr_fold.fit(X_clf_arr[tr_idx], y_noise[tr_idx])

        # Train metrics
        y_pred_train = pipe_lr_fold.predict(X_clf_arr[tr_idx])
        train_acc = accuracy_score(y_noise[tr_idx], y_pred_train)
        lr_train_accs.append(train_acc)

        train_auc = np.nan
        try:
            y_prob_train = pipe_lr_fold.predict_proba(X_clf_arr[tr_idx])[:, 1]
            train_auc = safe_auc(y_noise[tr_idx], y_prob_train)
        except Exception:
            pass
        lr_train_aucs.append(train_auc)

        # Test metrics
        y_pred_test = pipe_lr_fold.predict(X_clf_arr[va_idx])
        test_acc = accuracy_score(y_noise[va_idx], y_pred_test)
        lr_test_accs.append(test_acc)

        test_auc = np.nan
        try:
            y_prob_test = pipe_lr_fold.predict_proba(X_clf_arr[va_idx])[:, 1]
            test_auc = safe_auc(y_noise[va_idx], y_prob_test)
        except Exception:
            pass
        lr_test_aucs.append(test_auc)

        noise_lr_train_test_rows.append({
            'scenario': 'noise_only',
            'classifier': 'LogisticRegression',
            'fold': fold_idx,
            'n_train': int(len(tr_idx)),
            'n_test': int(len(va_idx)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_roc_auc': float(train_auc) if not np.isnan(train_auc) else np.nan,
            'test_roc_auc': float(test_auc) if not np.isnan(test_auc) else np.nan,
        })
        fold_idx += 1

    noise_lr_cv_summary = {
        'scenario': 'noise_only',
        'classifier': 'LogisticRegression',
        'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
        'n_splits': int(n_splits_clf_noise),
        'train_accuracy_mean': float(np.mean(lr_train_accs)) if lr_train_accs else np.nan,
        'train_accuracy_std': float(np.std(lr_train_accs)) if lr_train_accs else np.nan,
        'test_accuracy_mean': float(np.mean(lr_test_accs)) if lr_test_accs else np.nan,
        'test_accuracy_std': float(np.std(lr_test_accs)) if lr_test_accs else np.nan,
        'train_roc_auc_mean': float(np.nanmean(lr_train_aucs)) if lr_train_aucs else np.nan,
        'train_roc_auc_std': float(np.nanstd(lr_train_aucs)) if lr_train_aucs else np.nan,
        'test_roc_auc_mean': float(np.nanmean(lr_test_aucs)) if lr_test_aucs else np.nan,
        'test_roc_auc_std': float(np.nanstd(lr_test_aucs)) if lr_test_aucs else np.nan,
    }

    # --- Random Forest ---
    print("    [NOISE-ONLY][RF] Random Forest …")
    pipe_rf_noise = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                        class_weight='balanced',
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1))
    ])
    rf_train_accs = []
    rf_test_accs = []
    rf_train_aucs = []
    rf_test_aucs = []
    fold_idx = 0
    for tr_idx, va_idx in cv_clf_noise.split(X_clf_arr, y_noise):
        pipe_rf_fold = clone(pipe_rf_noise)
        pipe_rf_fold.fit(X_clf_arr[tr_idx], y_noise[tr_idx])

        # Train metrics
        y_pred_train = pipe_rf_fold.predict(X_clf_arr[tr_idx])
        train_acc = accuracy_score(y_noise[tr_idx], y_pred_train)
        rf_train_accs.append(train_acc)

        train_auc = np.nan
        try:
            y_prob_train = pipe_rf_fold.predict_proba(X_clf_arr[tr_idx])[:, 1]
            train_auc = safe_auc(y_noise[tr_idx], y_prob_train)
        except Exception:
            pass
        rf_train_aucs.append(train_auc)

        # Test metrics
        y_pred_test = pipe_rf_fold.predict(X_clf_arr[va_idx])
        test_acc = accuracy_score(y_noise[va_idx], y_pred_test)
        rf_test_accs.append(test_acc)

        test_auc = np.nan
        try:
            y_prob_test = pipe_rf_fold.predict_proba(X_clf_arr[va_idx])[:, 1]
            test_auc = safe_auc(y_noise[va_idx], y_prob_test)
        except Exception:
            pass
        rf_test_aucs.append(test_auc)

        noise_rf_train_test_rows.append({
            'scenario': 'noise_only',
            'classifier': 'RandomForest',
            'fold': fold_idx,
            'n_train': int(len(tr_idx)),
            'n_test': int(len(va_idx)),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_roc_auc': float(train_auc) if not np.isnan(train_auc) else np.nan,
            'test_roc_auc': float(test_auc) if not np.isnan(test_auc) else np.nan,
        })
        fold_idx += 1

    noise_rf_cv_summary = {
        'scenario': 'noise_only',
        'classifier': 'RandomForest',
        'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
        'n_splits': int(n_splits_clf_noise),
        'train_accuracy_mean': float(np.mean(rf_train_accs)) if rf_train_accs else np.nan,
        'train_accuracy_std': float(np.std(rf_train_accs)) if rf_train_accs else np.nan,
        'test_accuracy_mean': float(np.mean(rf_test_accs)) if rf_test_accs else np.nan,
        'test_accuracy_std': float(np.std(rf_test_accs)) if rf_test_accs else np.nan,
        'train_roc_auc_mean': float(np.nanmean(rf_train_aucs)) if rf_train_aucs else np.nan,
        'train_roc_auc_std': float(np.nanstd(rf_train_aucs)) if rf_train_aucs else np.nan,
        'test_roc_auc_mean': float(np.nanmean(rf_test_aucs)) if rf_test_aucs else np.nan,
        'test_roc_auc_std': float(np.nanstd(rf_test_aucs)) if rf_test_aucs else np.nan,
    }
else:
    print("    [NOISE-ONLY] Skipped CV (too few samples per class)")

# Fit on full data for importances
pipe_rf.fit(X_clf_arr, y_clf_arr)
imp_rf = pipe_rf.named_steps['clf'].feature_importances_
rf_importance = pd.DataFrame({
    'Feature':    selected_features,
    'RF_Importance': imp_rf
}).sort_values('RF_Importance', ascending=False).reset_index(drop=True)

# Merge both into one table
clf_importance = lr_importance.merge(
    rf_importance[['Feature', 'RF_Importance']], on='Feature', how='outer'
).sort_values('LR_Abs', ascending=False).reset_index(drop=True)

# =============================================================================
# ── STEP 7 : Plots ────────────────────────────────────────────────────────────
# =============================================================================
print(f"\n[7] Saving plots …")

# ── 7a. PCA scatter coloured by cluster ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Colour map: -1 (noise) = grey, clusters = tab10
cmap   = plt.get_cmap('tab10')
all_c  = sorted(set(cluster_labels))
colors = {c: ('lightgrey' if c == -1 else cmap(i % 10))
          for i, c in enumerate(all_c)}
point_colors = [colors[c] for c in cluster_labels]

ax = axes[0]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors,
           s=40, alpha=0.8, edgecolors='k', linewidths=0.3)
ax.set_xlabel('PC 1', fontsize=11)
ax.set_ylabel('PC 2', fontsize=11)
ax.set_title('HDBSCAN Clusters (PCA space)', fontsize=13, fontweight='bold')
patches = [mpatches.Patch(color=colors[c],
                           label=f'Cluster {c}' if c >= 0 else 'Noise (-1)')
           for c in all_c]
ax.legend(handles=patches, fontsize=8, loc='best')

# ── 7b. PCA scatter coloured by retained / discarded ─────────────────────────
ax2 = axes[1]
point_colors2 = ['#2196F3' if r else '#F44336' for r in retained_mask]
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors2,
            s=40, alpha=0.8, edgecolors='k', linewidths=0.3)
ax2.set_xlabel('PC 1', fontsize=11)
ax2.set_ylabel('PC 2', fontsize=11)
ax2.set_title('Retained vs. Discarded Samples (PCA space)', fontsize=13, fontweight='bold')
ax2.legend(handles=[
    mpatches.Patch(color='#2196F3', label=f'Retained (n={n_retained})'),
    mpatches.Patch(color='#F44336', label=f'Discarded (n={n_discarded})')
], fontsize=10)

plt.tight_layout()
pca_plot = os.path.join(OUTPUT_DIR, 'cluster_pca_scatter.png')
plt.savefig(pca_plot, dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved: {pca_plot}")

# ── 7c. CV predictions scatter (retained subset) ─────────────────────────────
y_ret_vals  = y_ret.values
y_pred_ret  = (results_retained['predictions'].values
               if isinstance(results_retained['predictions'], pd.Series)
               else results_retained['predictions'])
y_full_vals = y.values
y_pred_full = (results_full['predictions'].values
               if isinstance(results_full['predictions'], pd.Series)
               else results_full['predictions'])

if LOG_TRANSFORM:
    # Top row: log(1 + days) space (current behaviour)
    # Bottom row: back-transform to original Survival Days using expm1
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ----- Top row: log space -----
    for ax_i, (yv, yp, title, r2v, corr) in enumerate([
        (y_full_vals, y_pred_full, 'Full Dataset',
         results_full['mean_score'], results_full['correlation']),
        (y_ret_vals,  y_pred_ret,  'Retained Subset',
         results_retained['mean_score'], results_retained['correlation'])
    ]):
        ax = axes[0, ax_i]
        ax.scatter(yv, yp, alpha=0.6, s=50,
                   color='steelblue' if ax_i == 0 else 'seagreen',
                   edgecolors='k', linewidths=0.3)
        lo, hi = min(yv.min(), yp.min()), max(yv.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='y = x')
        ax.set_xlabel(f'Actual ({target_label})', fontsize=11)
        ax.set_ylabel(f'Predicted ({target_label})', fontsize=11)
        ax.set_title(f'Log scale: Ridge CV — {title}\nR²={r2v:.4f}  Corr={corr:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ----- Bottom row: original days -----
    y_full_days = np.expm1(y_full_vals)
    y_ret_days = np.expm1(y_ret_vals)
    y_pred_full_days = np.expm1(y_pred_full)
    y_pred_ret_days = np.expm1(y_pred_ret)

    for ax_i, (yv_d, yp_d, title) in enumerate([
        (y_full_days, y_pred_full_days, 'Full Dataset'),
        (y_ret_days,  y_pred_ret_days,  'Retained Subset')
    ]):
        ax = axes[1, ax_i]
        ax.scatter(yv_d, yp_d, alpha=0.6, s=50,
                   color='steelblue' if ax_i == 0 else 'seagreen',
                   edgecolors='k', linewidths=0.3)
        lo, hi = min(yv_d.min(), yp_d.min()), max(yv_d.max(), yp_d.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='y = x')
        ax.set_xlabel('Actual (Survival Days)', fontsize=11)
        ax.set_ylabel('Predicted (Survival Days)', fontsize=11)
        ax.set_title(f'Original scale (expm1): Ridge CV — {title}\nR² & Corr computed in log space',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (yv, yp, title, r2v, corr) in enumerate([
        (y_full_vals, y_pred_full, 'Full Dataset',
         results_full['mean_score'], results_full['correlation']),
        (y_ret_vals,  y_pred_ret,  'Retained Subset',
         results_retained['mean_score'], results_retained['correlation'])
    ]):
        ax = axes[ax_i]
        ax.scatter(yv, yp, alpha=0.6, s=50,
                   color='steelblue' if ax_i == 0 else 'seagreen',
                   edgecolors='k', linewidths=0.3)
        lo, hi = min(yv.min(), yp.min()), max(yv.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='y = x')
        ax.set_xlabel(f'Actual ({target_label})', fontsize=11)
        ax.set_ylabel(f'Predicted ({target_label})', fontsize=11)
        ax.set_title(f'Ridge CV — {title}\nR²={r2v:.4f}  Corr={corr:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
cv_plot = os.path.join(OUTPUT_DIR, 'ridge_cv_full_vs_retained.png')
plt.savefig(cv_plot, dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved: {cv_plot}")

# ── 7d. Top-N feature importance: LR coefficients ────────────────────────────
top_lr = lr_importance.head(TOP_N_FEATURES_SHOW)
fig, ax = plt.subplots(figsize=(10, 7))
colors_lr = ['#D32F2F' if v < 0 else '#1565C0' for v in top_lr['LR_Coefficient']]
ax.barh(range(len(top_lr)), top_lr['LR_Coefficient'], color=colors_lr, alpha=0.8)
ax.set_yticks(range(len(top_lr)))
ax.set_yticklabels(top_lr['Feature'], fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('LR Coefficient  (+ → predicts retained)', fontsize=11)
ax.set_title(f'Top {TOP_N_FEATURES_SHOW} Genes — Logistic Regression\n'
             f'(separating retained vs. discarded samples)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
lr_plot = os.path.join(OUTPUT_DIR, 'classifier_lr_feature_importance.png')
plt.savefig(lr_plot, dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved: {lr_plot}")

# ── 7e. Top-N feature importance: RF ─────────────────────────────────────────
top_rf = rf_importance.head(TOP_N_FEATURES_SHOW)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(top_rf)), top_rf['RF_Importance'], color='#2E7D32', alpha=0.8)
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf['Feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Random Forest Feature Importance (Gini)', fontsize=11)
ax.set_title(f'Top {TOP_N_FEATURES_SHOW} Genes — Random Forest\n'
             f'(separating retained vs. discarded samples)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
rf_plot = os.path.join(OUTPUT_DIR, 'classifier_rf_feature_importance.png')
plt.savefig(rf_plot, dpi=200, bbox_inches='tight')
plt.close()
print(f"    Saved: {rf_plot}")

# ── 7f. Cluster removal trace (score vs. step) ───────────────────────────────
if removal_log:
    steps   = [0]        + [d['step']        for d in removal_log]
    r2s     = [baseline_r2]  + [d['r2_after']    for d in removal_log]
    pscores = [baseline_score] + [d['penalised_score'] for d in removal_log]
    nkept   = [n_total]  + [d['n_kept']      for d in removal_log]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(steps, r2s,     'b-o', lw=2, ms=7, label='CV R²')
    l2, = ax1.plot(steps, pscores, 'g--s', lw=2, ms=7, label=f'Penalised score (λ={LAMBDA_PENALTY})')
    l3, = ax2.plot(steps, nkept,   'r:^', lw=2, ms=7, label='N retained')
    ax1.set_xlabel('Removal step', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11, color='navy')
    ax2.set_ylabel('N samples retained', fontsize=11, color='darkred')
    ax1.set_title('Cluster removal trace', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax2.tick_params(axis='y', labelcolor='darkred')
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    trace_plot = os.path.join(OUTPUT_DIR, 'cluster_removal_trace.png')
    plt.savefig(trace_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {trace_plot}")

# =============================================================================
# ── STEP 8 : Save CSVs ────────────────────────────────────────────────────────
# =============================================================================
print(f"\n[8] Saving CSV outputs …")

# Sample membership
membership_df = pd.DataFrame({
    'SampleID':      X.index,
    'Cluster':       cluster_labels,
    'Retained':      retained_mask.astype(int),
    'Survival_target': y.values
}).set_index('SampleID')
f = os.path.join(OUTPUT_DIR, 'sample_cluster_membership.csv')
membership_df.to_csv(f)
print(f"    {f}")

# Cluster-level stats
cluster_df = pd.DataFrame([
    {'Cluster': cid,
     'N_samples': cluster_stats[cid]['n'],
     'CV_R2': cluster_stats[cid]['r2'],
     'Retained': cid in active_clusters}
    for cid in unique_clusters
])
f = os.path.join(OUTPUT_DIR, 'cluster_statistics.csv')
cluster_df.to_csv(f, index=False)
print(f"    {f}")

# Removal log
if removal_log:
    f = os.path.join(OUTPUT_DIR, 'cluster_removal_log.csv')
    pd.DataFrame(removal_log).to_csv(f, index=False)
    print(f"    {f}")

# Classifier importances
f = os.path.join(OUTPUT_DIR, 'classifier_feature_importance.csv')
clf_importance.to_csv(f, index=False)
print(f"    {f}")

# Selected features
f = os.path.join(OUTPUT_DIR, 'selected_features.csv')
pd.DataFrame({'Feature': selected_features}).to_csv(f, index=False)
print(f"    {f}")

# ── Classifier performance metrics (CV + train/test) ───────────────────────
features_file = os.path.join(OUTPUT_DIR, 'classifier_features_used_for_cv.csv')
pd.DataFrame([{
    'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
    'n_cv_splits_classifier': int(n_splits_clf),
}]).to_csv(features_file, index=False)
print(f"    {features_file}")

if lr_train_test_rows:
    lr_rows_file = os.path.join(OUTPUT_DIR, 'classifier_lr_train_test_metrics.csv')
    pd.DataFrame(lr_train_test_rows).to_csv(lr_rows_file, index=False)
    print(f"    {lr_rows_file}")

    lr_summary_file = os.path.join(OUTPUT_DIR, 'classifier_lr_cv_summary.csv')
    pd.DataFrame([lr_cv_summary]).to_csv(lr_summary_file, index=False)
    print(f"    {lr_summary_file}")

if rf_train_test_rows:
    rf_rows_file = os.path.join(OUTPUT_DIR, 'classifier_rf_train_test_metrics.csv')
    pd.DataFrame(rf_train_test_rows).to_csv(rf_rows_file, index=False)
    print(f"    {rf_rows_file}")

    rf_summary_file = os.path.join(OUTPUT_DIR, 'classifier_rf_cv_summary.csv')
    pd.DataFrame([rf_cv_summary]).to_csv(rf_summary_file, index=False)
    print(f"    {rf_summary_file}")

# Noise-only classifier metrics
if noise_lr_train_test_rows:
    noise_features_file = os.path.join(OUTPUT_DIR, 'classifier_features_used_for_cv_noise_only.csv')
    pd.DataFrame([{
        'n_features_used_for_classifier_cv': n_features_used_for_classifier_cv,
    }]).to_csv(noise_features_file, index=False)
    print(f"    {noise_features_file}")

    noise_lr_rows_file = os.path.join(OUTPUT_DIR, 'classifier_lr_train_test_metrics_noise_only.csv')
    pd.DataFrame(noise_lr_train_test_rows).to_csv(noise_lr_rows_file, index=False)
    print(f"    {noise_lr_rows_file}")

    noise_lr_summary_file = os.path.join(OUTPUT_DIR, 'classifier_lr_cv_summary_noise_only.csv')
    pd.DataFrame([noise_lr_cv_summary]).to_csv(noise_lr_summary_file, index=False)
    print(f"    {noise_lr_summary_file}")

if noise_rf_train_test_rows:
    noise_rf_rows_file = os.path.join(OUTPUT_DIR, 'classifier_rf_train_test_metrics_noise_only.csv')
    pd.DataFrame(noise_rf_train_test_rows).to_csv(noise_rf_rows_file, index=False)
    print(f"    {noise_rf_rows_file}")

    noise_rf_summary_file = os.path.join(OUTPUT_DIR, 'classifier_rf_cv_summary_noise_only.csv')
    pd.DataFrame([noise_rf_cv_summary]).to_csv(noise_rf_summary_file, index=False)
    print(f"    {noise_rf_summary_file}")

# =============================================================================
# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Total samples          : {n_total}")
print(f"  HDBSCAN clusters       : {n_clusters}  (+{n_noise} noise points)")
print(f"  Clusters removed       : {len(removal_log)}")
print(f"  Retained samples       : {n_retained}  ({100*n_retained/n_total:.1f}%)")
print(f"  Discarded samples      : {n_discarded}  ({100*n_discarded/n_total:.1f}%)")
print()
print(f"  Ridge CV R²  (full)    : {results_full['mean_score']:.4f} ± "
      f"{results_full['std_score']:.4f}")
print(f"  Ridge CV R²  (retained): {results_retained['mean_score']:.4f} ± "
      f"{results_retained['std_score']:.4f}")
print(f"  Δ R²                   : {results_retained['mean_score'] - results_full['mean_score']:+.4f}")
print(f"  Corr (full)            : {results_full['correlation']:.4f}")
print(f"  Corr (retained)        : {results_retained['correlation']:.4f}")
print()
if len(acc_lr):
    print(f"  LR classifier  Acc={np.mean(acc_lr):.4f}  "
          f"AUC={np.mean(auc_lr):.4f}")
if len(acc_rf):
    print(f"  RF classifier  Acc={np.mean(acc_rf):.4f}  "
          f"AUC={np.mean(auc_rf):.4f}")
print()
print(f"  Top 5 LR genes  : {lr_importance['Feature'].head(5).tolist()}")
print(f"  Top 5 RF genes  : {rf_importance['Feature'].head(5).tolist()}")
print("=" * 65)
