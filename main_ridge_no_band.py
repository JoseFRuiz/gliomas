# Ridge Regression Performance Evaluation (no band filtering)
# Same as main_ridge_only.py but without filtering samples by prediction-band percentile

from utils import load_data, cross_validate_regression, filter_samples_for_model_with_features
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data and options
gene_tpm_path = os.path.join('data', 'TCGAGliomas_RNAm_Filtrado_QC_verif.csv')
log_transform_survival = True  # If True, model and metrics use y = log(1 + survival_days)

X, y = load_data(gene_tpm_path=gene_tpm_path)

# Optional: log-transform survival days (log(1 + x)) for modeling
if log_transform_survival:
    if isinstance(y, pd.Series):
        y = pd.Series(np.log1p(y.values), index=y.index, name=y.name)
    else:
        y = np.log1p(y)

# Set up parameters
n_samples = X.shape[0]
max_outliers = int(n_samples * 0.5)
n_features = 1000  # Number of features to select
ridge_alpha = 100.0  # Regularization strength

print("="*60)
print("RIDGE REGRESSION PERFORMANCE (NO BAND FILTERING)")
print("="*60)
print(f"Total samples: {n_samples}")
print(f"Max outliers to remove: {max_outliers}")
print(f"Number of features to select: {n_features}")
print(f"Ridge alpha (regularization): {ridge_alpha}")
print(f"Log-transform survival: {log_transform_survival}")
if log_transform_survival:
    y_vals = y.values if isinstance(y, pd.Series) else y
    print(f"  (target = log(1+days); range [{y_vals.min():.2f}, {y_vals.max():.2f}])")
print("="*60)

# Step 1: Filter samples and select features for Ridge
print("\n--- Step 1: Filtering samples and selecting features for Ridge ---")
ridge_model = Ridge(alpha=ridge_alpha)
X_ridge_filtered, y_ridge_filtered, kept_samples_ridge, removed_ridge, selected_features_ridge = filter_samples_for_model_with_features(
    X, y, ridge_model, n_features=n_features, max_outliers_to_remove=max_outliers,
    min_improvement=0.01, random_state=42, model_name="Ridge"
)
print(f"  Samples selected: {len(kept_samples_ridge)}")
print(f"  Features selected: {len(selected_features_ridge)}")

# Step 2: Evaluate Ridge on filtered dataset
print("\n--- Step 2: Evaluating Ridge on filtered dataset ---")
results_ridge_filtered = cross_validate_regression(X_ridge_filtered, y_ridge_filtered,
                                                    model=ridge_model, model_name='Ridge (Filtered)')
print(f"  R² (CV): {results_ridge_filtered['mean_score']:.4f} ± {results_ridge_filtered['std_score']:.4f}")
print(f"  Correlation: {results_ridge_filtered['correlation']:.4f}")

# Step 3: Evaluate Ridge on FULL dataset with selected features
print("\n--- Step 3: Evaluating Ridge on FULL dataset with selected features ---")
if isinstance(X, pd.DataFrame):
    X_full_ridge = X[selected_features_ridge]
else:
    feature_indices = [X.columns.get_loc(f) for f in selected_features_ridge] if isinstance(X, pd.DataFrame) else selected_features_ridge
    X_full_ridge = X[:, feature_indices]

results_ridge_full = cross_validate_regression(X_full_ridge, y, model=ridge_model, model_name='Ridge (Full Dataset)')
print(f"  R² (CV): {results_ridge_full['mean_score']:.4f} ± {results_ridge_full['std_score']:.4f}")
print(f"  Correlation: {results_ridge_full['correlation']:.4f}")

# Step 4: Get cross-validation predictions for visualization
print("\n--- Step 4: Generating cross-validation predictions ---")
X_full_ridge_values = X_full_ridge.values if isinstance(X_full_ridge, pd.DataFrame) else X_full_ridge
y_values = y.values if isinstance(y, pd.Series) else y

survival_label = 'log(1 + Survival Days)' if log_transform_survival else 'Survival Days'

if len(y_values) >= 10:
    n_splits = min(5, max(3, len(y_values) // 3))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ridge_model_cv = Ridge(alpha=ridge_alpha)
    y_pred_ridge = cross_val_predict(ridge_model_cv, X_full_ridge_values, y_values, cv=cv)

    if isinstance(y, pd.Series):
        y_pred_ridge = pd.Series(y_pred_ridge, index=y.index)

    r2_ridge = r2_score(y_values, y_pred_ridge)
    corr_ridge = pearsonr(y_values, y_pred_ridge)[0] if len(y_values) > 1 else 0

    print(f"  R²: {r2_ridge:.4f}")
    print(f"  Correlation: {corr_ridge:.4f}")

    # Step 5: Visualize predictions (log scale and, if transformed, original scale)
    print("\n--- Step 5: Creating visualization ---")
    y_pred_vals = y_pred_ridge.values if isinstance(y_pred_ridge, pd.Series) else y_pred_ridge

    if log_transform_survival:
        fig, (ax_log, ax_orig) = plt.subplots(1, 2, figsize=(16, 8))
        ax_log.scatter(y_values, y_pred_vals, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.5)
        ax_log.plot([y_values.min(), y_values.max()],
                    [y_values.min(), y_values.max()], 'r--', lw=2, label='Perfect prediction')
        ax_log.set_xlabel(f'Actual ({survival_label})', fontsize=12, fontweight='bold')
        ax_log.set_ylabel(f'Predicted ({survival_label})', fontsize=12, fontweight='bold')
        ax_log.set_title(f'Log scale (model space)\nR² = {r2_ridge:.4f}, Corr = {corr_ridge:.4f}', fontsize=13, fontweight='bold', pad=10)
        ax_log.legend(fontsize=11)
        ax_log.grid(True, alpha=0.3)
        y_act_days = np.expm1(y_values)
        y_pred_days = np.expm1(y_pred_vals)
        ax_orig.scatter(y_act_days, y_pred_days, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.5)
        ax_orig.plot([y_act_days.min(), y_act_days.max()],
                     [y_act_days.min(), y_act_days.max()], 'r--', lw=2, label='Perfect prediction')
        ax_orig.set_xlabel('Actual (Survival Days)', fontsize=12, fontweight='bold')
        ax_orig.set_ylabel('Predicted (Survival Days)', fontsize=12, fontweight='bold')
        ax_orig.set_title('Original scale (expm1 of log-scale predictions)\nR² and Corr are in log space', fontsize=13, fontweight='bold', pad=10)
        ax_orig.legend(fontsize=11)
        ax_orig.grid(True, alpha=0.3)
    else:
        fig, ax_log = plt.subplots(1, 1, figsize=(10, 8))
        ax_log.scatter(y_values, y_pred_vals, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.5)
        ax_log.plot([y_values.min(), y_values.max()],
                    [y_values.min(), y_values.max()], 'r--', lw=2, label='Perfect prediction')
        ax_log.set_xlabel(f'Actual Values ({survival_label})', fontsize=12, fontweight='bold')
        ax_log.set_ylabel(f'Predicted Values ({survival_label})', fontsize=12, fontweight='bold')
        ax_log.set_title(f'Ridge Performance\n(R² = {r2_ridge:.4f}, Correlation = {corr_ridge:.4f})', fontsize=14, fontweight='bold', pad=15)
        ax_log.legend(fontsize=11)
        ax_log.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ridge_performance_no_band.png', dpi=300, bbox_inches='tight')
    print("  Saved plot to: ridge_performance_no_band.png")
    plt.show()

else:
    print("  WARNING: Insufficient samples for cross-validation predictions")
    y_pred_ridge = None

# Step 6: Feature Relevance Analysis
print("\n--- Step 6: Feature Relevance Analysis ---")

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

sample_index = y.index if isinstance(y, pd.Series) else np.arange(len(y))
ridge_kept_ids = [sample_index[i] for i in kept_samples_ridge]
ridge_removed_ids = [sample_index[i] for i in removed_ridge]
pd.DataFrame({'Sample_ID': ridge_kept_ids}).to_csv(os.path.join(output_dir, 'ridge_kept_samples.csv'), index=False)
pd.DataFrame({'Sample_ID': ridge_removed_ids}).to_csv(os.path.join(output_dir, 'ridge_removed_samples.csv'), index=False)
print(f"  Saved Ridge sample lists: {len(ridge_kept_ids)} kept, {len(ridge_removed_ids)} removed")

# Step 5b: Classifier to separate kept vs removed samples (using ALL features)
print("\n--- Step 5b: Classifier (kept vs removed) with all features ---")
# Binary labels: 1 = ridge kept, 0 = ridge removed (by integer position)
y_binary = np.array([1 if i in kept_samples_ridge else 0 for i in range(n_samples)])
X_all = X.values if isinstance(X, pd.DataFrame) else X
all_feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"F{i}" for i in range(X_all.shape[1])]

n_kept = y_binary.sum()
n_removed = len(y_binary) - n_kept
print(f"  Class balance: {n_kept} kept, {n_removed} removed")

pipe_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
])
n_splits_clf = min(5, n_kept, n_removed) if n_kept and n_removed else 0
perf_rows = []
if n_splits_clf >= 2:
    cv_clf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe_clf, X_all, y_binary, cv=cv_clf, scoring='accuracy', n_jobs=-1)
    acc_mean, acc_std = float(acc_scores.mean()), float(acc_scores.std())
    print(f"  Accuracy (CV): {acc_mean:.4f} ± {acc_std:.4f}")
    perf_rows.append({'metric': 'accuracy', 'mean': acc_mean, 'std': acc_std, 'n_folds': len(acc_scores)})
    auc_mean, auc_std = np.nan, np.nan
    try:
        auc_scores = cross_val_score(pipe_clf, X_all, y_binary, cv=cv_clf, scoring='roc_auc', n_jobs=-1)
        auc_mean, auc_std = float(auc_scores.mean()), float(auc_scores.std())
        print(f"  ROC-AUC (CV): {auc_mean:.4f} ± {auc_std:.4f}")
        perf_rows.append({'metric': 'roc_auc', 'mean': auc_mean, 'std': auc_std, 'n_folds': len(auc_scores)})
    except Exception:
        print("  ROC-AUC: not computed (e.g. one class in a fold)")
        perf_rows.append({'metric': 'roc_auc', 'mean': np.nan, 'std': np.nan, 'n_folds': 0})
else:
    print("  Skipping CV (need at least 2 samples per class for stratified folds).")

# Save classification performance to CSV
if perf_rows:
    perf_df = pd.DataFrame(perf_rows)
    perf_file = os.path.join(output_dir, 'classifier_kept_vs_removed_performance.csv')
    perf_df.to_csv(perf_file, index=False)
    print(f"  Saved classification performance to: {perf_file}")

# Fit on full data and save feature coefficients (which features best separate kept vs removed)
pipe_clf.fit(X_all, y_binary)
coef_clf = pipe_clf.named_steps['clf'].coef_.ravel()
clf_weights_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coef_clf,
    'Abs_Coefficient': np.abs(coef_clf)
}).sort_values('Abs_Coefficient', ascending=False)
clf_weights_file = os.path.join(output_dir, 'classifier_kept_vs_removed_all_features.csv')
clf_weights_df.to_csv(clf_weights_file, index=False)
print(f"  Saved classifier feature weights (all features) to: {clf_weights_file}")

# Step 5c: Exploration — try several approaches to improve kept vs removed separation
print("\n--- Step 5c: Exploration (improve kept vs removed classifier) ---")
exploration_results = []
cv_clf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True, random_state=42) if n_splits_clf >= 2 else None

def _eval_clf(pipe, X_in, name, extra_note=""):
    """Run CV and append to exploration_results."""
    if cv_clf is None:
        return
    acc = cross_val_score(pipe, X_in, y_binary, cv=cv_clf, scoring='accuracy', n_jobs=-1)
    row = {'approach': name, 'accuracy_mean': acc.mean(), 'accuracy_std': acc.std(), 'roc_auc_mean': np.nan, 'roc_auc_std': np.nan, 'notes': extra_note}
    try:
        auc = cross_val_score(pipe, X_in, y_binary, cv=cv_clf, scoring='roc_auc', n_jobs=-1)
        row['roc_auc_mean'], row['roc_auc_std'] = auc.mean(), auc.std()
    except Exception:
        pass
    exploration_results.append(row)
    print(f"  {name}: Acc = {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}, ROC-AUC = {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}  {extra_note}")

if cv_clf is not None:
    # 1) Meta-features: Ridge OOF prediction + |residual| (why a sample was removed may relate to fit)
    if len(y_values) >= 10:
        ridge_oof = cross_val_predict(Ridge(alpha=ridge_alpha), X_full_ridge_values, y_values, cv=KFold(n_splits=min(5, max(3, len(y_values)//3)), shuffle=True, random_state=42))
        resid = y_values - ridge_oof
        abs_resid = np.abs(resid)
        X_meta = np.column_stack([ridge_oof, abs_resid])
        pipe_meta = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))])
        _eval_clf(pipe_meta, X_meta, "meta_features (Ridge pred + |residual|)", "(2 features)")

        # 1b) Extended meta-features + different classifiers (to improve meta-features performance)
        rel_resid = resid / (np.abs(y_values) + 1e-6)
        X_meta_ext = np.column_stack([ridge_oof, abs_resid, resid, resid**2, rel_resid])
        pipe_meta_ext = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))])
        _eval_clf(pipe_meta_ext, X_meta_ext, "meta_ext (pred,|res|,res,res²,rel_res) + LogReg", "(5 features)")
        pipe_meta_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42))])
        _eval_clf(pipe_meta_rf, X_meta, "meta_features (2 feat) + RandomForest", "(2 features)")
        pipe_meta_ext_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42))])
        _eval_clf(pipe_meta_ext_rf, X_meta_ext, "meta_ext (5 feat) + RandomForest", "(5 features)")
        pipe_meta_svc = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42, probability=True))])
        _eval_clf(pipe_meta_svc, X_meta, "meta_features (2 feat) + SVM-RBF", "(2 features)")
        pipe_meta_ext_svc = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42, probability=True))])
        _eval_clf(pipe_meta_ext_svc, X_meta_ext, "meta_ext (5 feat) + SVM-RBF", "(5 features)")

    # 2) L1 Logistic Regression (sparse, may focus on informative genes)
    pipe_l1 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=2000, random_state=42, class_weight='balanced'))
    ])
    _eval_clf(pipe_l1, X_all, "L1_LogReg (C=0.1, all features)")
    pipe_l1.fit(X_all, y_binary)
    nz = np.sum(np.abs(pipe_l1.named_steps['clf'].coef_) > 1e-5)
    exploration_results[-1]['notes'] = f"non-zero coefs: {nz}"

    # 3) Univariate feature selection (top k by F-stat) + L2 LogReg
    k_sel = min(500, X_all.shape[1], X_all.shape[0] // 3)
    if k_sel >= 10:
        pipe_sel = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_sel)),
            ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
        ])
        _eval_clf(pipe_sel, X_all, f"SelectKBest(k={k_sel}) + LogReg", f"({k_sel} features)")

    # 4) Random Forest (often robust in high-dim)
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42))
    ])
    _eval_clf(pipe_rf, X_all, "RandomForest (all features)")

    # Save exploration summary
    if exploration_results:
        exp_df = pd.DataFrame(exploration_results)
        exp_file = os.path.join(output_dir, 'classifier_kept_vs_removed_exploration.csv')
        exp_df.to_csv(exp_file, index=False)
        print(f"  Saved exploration comparison to: {exp_file}")
        best_auc = exp_df['roc_auc_mean'].max()
        if not np.isnan(best_auc):
            best_row = exp_df.loc[exp_df['roc_auc_mean'].idxmax()]
            print(f"  Best ROC-AUC in exploration: {best_row['approach']} ({best_auc:.3f})")
        else:
            print("  No valid ROC-AUC in exploration.")

ridge_final = Ridge(alpha=ridge_alpha)
ridge_final.fit(X_full_ridge_values, y_values)

coefficients = ridge_final.coef_
feature_importance_df = pd.DataFrame({
    'Feature': selected_features_ridge,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

feature_correlations = []
for feature in selected_features_ridge:
    if isinstance(X_full_ridge, pd.DataFrame):
        feature_values = X_full_ridge[feature].values
    else:
        feature_idx = selected_features_ridge.index(feature)
        feature_values = X_full_ridge_values[:, feature_idx]
    corr, _ = pearsonr(feature_values, y_values)
    feature_correlations.append(corr)

feature_importance_df['Correlation_with_Target'] = feature_correlations
feature_importance_df['Abs_Correlation'] = np.abs(feature_correlations)
feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)

print(f"  Top 10 features by absolute coefficient:")
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"    {row['Feature']}: Coef={row['Coefficient']:.4f}, Corr={row['Correlation_with_Target']:.4f}")

print("\n  Creating feature importance visualizations...")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

top_n = min(20, len(feature_importance_df))
top_features_coef = feature_importance_df.head(top_n)
colors_coef = ['red' if x < 0 else 'blue' for x in top_features_coef['Coefficient']]
ax1.barh(range(len(top_features_coef)), top_features_coef['Coefficient'], color=colors_coef, alpha=0.7)
ax1.set_yticks(range(len(top_features_coef)))
ax1.set_yticklabels(top_features_coef['Feature'], fontsize=9)
ax1.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax1.set_title(f'Top {top_n} Features by Ridge Coefficient\n(Red=Negative, Blue=Positive)', fontsize=13, fontweight='bold', pad=10)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

feature_importance_df_corr = feature_importance_df.sort_values('Abs_Correlation', ascending=False)
top_features_corr = feature_importance_df_corr.head(top_n)
colors_corr = ['red' if x < 0 else 'blue' for x in top_features_corr['Correlation_with_Target']]
ax2.barh(range(len(top_features_corr)), top_features_corr['Correlation_with_Target'], color=colors_corr, alpha=0.7)
ax2.set_yticks(range(len(top_features_corr)))
ax2.set_yticklabels(top_features_corr['Feature'], fontsize=9)
ax2.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold')
ax2.set_title(f'Top {top_n} Features by Correlation with Target\n(Red=Negative, Blue=Positive)', fontsize=13, fontweight='bold', pad=10)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('ridge_feature_importance_no_band.png', dpi=300, bbox_inches='tight')
print("  Saved feature importance plot to: ridge_feature_importance_no_band.png")
plt.show()

importance_file = os.path.join(output_dir, 'ridge_feature_importance.csv')
feature_importance_df.to_csv(importance_file, index=False)
print(f"  Saved feature importance data to: {importance_file}")

print("\n  Feature Importance Statistics:")
print(f"    Mean absolute coefficient: {feature_importance_df['Abs_Coefficient'].mean():.4f}")
print(f"    Median absolute coefficient: {feature_importance_df['Abs_Coefficient'].median():.4f}")
print(f"    Max absolute coefficient: {feature_importance_df['Abs_Coefficient'].max():.4f}")
print(f"    Min absolute coefficient: {feature_importance_df['Abs_Coefficient'].min():.4f}")
print(f"    Mean absolute correlation: {feature_importance_df['Abs_Correlation'].mean():.4f}")
print(f"    Median absolute correlation: {feature_importance_df['Abs_Correlation'].median():.4f}")

# Step 7: Summary
print("\n" + "="*60)
print("RIDGE REGRESSION PERFORMANCE SUMMARY (NO BAND FILTERING)")
print("="*60)
print(f"\nFiltered Dataset ({len(kept_samples_ridge)} samples):")
print(f"  R² (CV): {results_ridge_filtered['mean_score']:.4f} ± {results_ridge_filtered['std_score']:.4f}")
print(f"  Correlation: {results_ridge_filtered['correlation']:.4f}")

print(f"\nFull Dataset ({len(y)} samples):")
print(f"  R² (CV): {results_ridge_full['mean_score']:.4f} ± {results_ridge_full['std_score']:.4f}")
print(f"  Correlation: {results_ridge_full['correlation']:.4f}")

if y_pred_ridge is not None:
    print(f"\nCross-Validation Predictions (Full Dataset):")
    print(f"  R²: {r2_ridge:.4f}")
    print(f"  Correlation: {corr_ridge:.4f}")

print(f"\nSelected Features: {len(selected_features_ridge)}")
print(f"Ridge Alpha (Regularization): {ridge_alpha}")
print("="*60)

features_file = os.path.join(output_dir, 'ridge_selected_features.csv')
features_df = pd.DataFrame({'Feature': selected_features_ridge})
features_df.to_csv(features_file, index=False)
print(f"\nSaved selected features to: {features_file}")
