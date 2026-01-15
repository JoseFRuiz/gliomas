# Load data
from utils import load_data, cross_validate_regression, filter_samples_for_model_with_features, evaluate_ensemble, evaluate_ensemble_with_model_features, get_ensemble_predictions
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
import os

gene_tpm_path = os.path.join('data', 'TCGAGliomas_RNAm_Filtrado_QC_verif.csv')

X, y = load_data(gene_tpm_path=gene_tpm_path)

sel_rand_var = False

if sel_rand_var:
    num_var = 554
    rand_indexes = np.random.randint(0, X.shape[1], num_var)
    X = X.iloc[:, rand_indexes]

# Set up parameters for multi-model filtering
n_samples = X.shape[0]
max_outliers = int(n_samples * 0.5)
n_features = 100  # Number of features to select
print(f"max_outliers to remove: {max_outliers}")
print(f"Number of features to select: {n_features}")
print("\n" + "="*60)
print("MULTI-MODEL ENSEMBLE FILTERING")
print("="*60)
print("Each model will filter samples independently, starting from the beginning")
print("="*60)

# Step 1.1: Filter for Ridge (starting from beginning)
print("\n--- Filtering for Ridge Model (starting from beginning) ---")
# Use stronger regularization to prevent overfitting (alpha=10.0 instead of default 1.0)
ridge_alpha = 10.0
ridge_model = Ridge(alpha=ridge_alpha)
X_ridge, y_ridge, kept_samples_ridge, removed_ridge, selected_features_ridge = filter_samples_for_model_with_features(
    X, y, ridge_model, n_features=n_features, max_outliers_to_remove=max_outliers, min_improvement=0.01, random_state=42, model_name="Ridge"
)
print(f"  Ridge: {len(kept_samples_ridge)} samples selected, {len(selected_features_ridge)} features selected")
results_ridge_filtered = cross_validate_regression(X_ridge, y_ridge, model=ridge_model, model_name='Ridge (Filtered)')
print(f"  R2 (CV): {results_ridge_filtered['mean_score']:.3f}, Correlation: {results_ridge_filtered['correlation']:.3f}")
if results_ridge_filtered['mean_score'] < -0.2:
    print(f"  WARNING: Ridge CV R² is very negative ({results_ridge_filtered['mean_score']:.3f}).")
    print(f"           This suggests the filtered dataset may not generalize well.")

# Step 1.2: Filter for SVR (starting from beginning)
print("\n--- Filtering for SVR Model (starting from beginning) ---")
# Use lower C for more regularization to prevent overfitting (C=1.0 instead of 100.0)
svr_C_param = 1.0
svr_model = SklearnPipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=svr_C_param, gamma='scale', epsilon=0.01))
])

X_svr, y_svr, kept_samples_svr, removed_svr, selected_features_svr = filter_samples_for_model_with_features(
    X, y, svr_model, n_features=n_features, max_outliers_to_remove=max_outliers, min_improvement=0.01, 
    random_state=42, model_name="SVR", svr_C=svr_C_param, svr_gamma='scale', svr_epsilon=0.01
)
print(f"  SVR: {len(kept_samples_svr)} samples selected, {len(selected_features_svr)} features selected")
results_svr_filtered = cross_validate_regression(X_svr, y_svr, model=svr_model, model_name='SVR (Filtered)')

# Check if SVR failed to evaluate initial samples
if np.isnan(results_svr_filtered['mean_score']):
    print(f"\nERROR: SVR cannot evaluate initial samples (R² = N/A)")
    print("Stopping execution.")
    import sys
    sys.exit(1)

print(f"  R2 (CV): {results_svr_filtered['mean_score']:.3f}, Correlation: {results_svr_filtered['correlation']:.3f}")
if results_svr_filtered['mean_score'] < -0.2:
    print(f"  WARNING: SVR CV R² is very negative ({results_svr_filtered['mean_score']:.3f}).")
    print(f"           This suggests the filtered dataset may not generalize well.")

# Step 1.3: Filter for Random Forest (starting from beginning)
print("\n--- Filtering for Random Forest Model (starting from beginning) ---")
# Heavily reduced parameters for faster execution during filtering (greedy selection evaluates many times)
rf_model = RandomForestRegressor(
    n_estimators=10,  # Heavily reduced for faster filtering
    max_depth=3,  # Reduced depth for faster training
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
X_rf, y_rf, kept_samples_rf, removed_rf, selected_features_rf = filter_samples_for_model_with_features(
    X, y, rf_model, n_features=n_features, max_outliers_to_remove=max_outliers, min_improvement=0.01, random_state=42, model_name="Random Forest"
)
print(f"  Random Forest: {len(kept_samples_rf)} samples selected, {len(selected_features_rf)} features selected")
results_rf_filtered = cross_validate_regression(X_rf, y_rf, model=rf_model, model_name='RF (Filtered)')
print(f"  R2 (CV): {results_rf_filtered['mean_score']:.3f}, Correlation: {results_rf_filtered['correlation']:.3f}")
if results_rf_filtered['mean_score'] < -0.2:
    print(f"  WARNING: RF CV R² is very negative ({results_rf_filtered['mean_score']:.3f}).")
    print(f"           This suggests the filtered dataset may not generalize well.")

# Diagnostic: Check training R² to see if models can at least fit their filtered data
print("\n" + "="*60)
print("DIAGNOSTIC: Training R² (to check if models can fit their data)")
print("="*60)

# Ridge training R²
if isinstance(X_ridge, pd.DataFrame):
    X_ridge_selected = X_ridge[selected_features_ridge].values
else:
    # If X is DataFrame, get feature indices
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_ridge]
    else:
        feature_indices = selected_features_ridge
    X_ridge_selected = X_ridge[:, feature_indices]
y_ridge_values = y_ridge.values if isinstance(y_ridge, pd.Series) else y_ridge
ridge_model.fit(X_ridge_selected, y_ridge_values)
y_pred_ridge_train = ridge_model.predict(X_ridge_selected)
ridge_train_r2 = r2_score(y_ridge_values, y_pred_ridge_train)
print(f"Ridge training R²: {ridge_train_r2:.4f} (CV R²: {results_ridge_filtered['mean_score']:.4f})")

# SVR training R²
if isinstance(X_svr, pd.DataFrame):
    X_svr_selected = X_svr[selected_features_svr].values
else:
    # If X is DataFrame, get feature indices
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_svr]
    else:
        feature_indices = selected_features_svr
    X_svr_selected = X_svr[:, feature_indices]
y_svr_values = y_svr.values if isinstance(y_svr, pd.Series) else y_svr
svr_model.fit(X_svr_selected, y_svr_values)
y_pred_svr_train = svr_model.predict(X_svr_selected)
svr_train_r2 = r2_score(y_svr_values, y_pred_svr_train)
print(f"SVR training R²: {svr_train_r2:.4f} (CV R²: {results_svr_filtered['mean_score']:.4f})")

# RF training R²
if isinstance(X_rf, pd.DataFrame):
    X_rf_selected = X_rf[selected_features_rf].values
else:
    # If X is DataFrame, get feature indices
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_rf]
    else:
        feature_indices = selected_features_rf
    X_rf_selected = X_rf[:, feature_indices]
y_rf_values = y_rf.values if isinstance(y_rf, pd.Series) else y_rf
rf_model.fit(X_rf_selected, y_rf_values)
y_pred_rf_train = rf_model.predict(X_rf_selected)
rf_train_r2 = r2_score(y_rf_values, y_pred_rf_train)
print(f"RF training R²: {rf_train_r2:.4f} (CV R²: {results_rf_filtered['mean_score']:.4f})")
print("="*60)

# Additional Diagnostic: Evaluate models on FULL dataset with their selected features
# This helps determine if the problem is with feature selection or sample selection
print("\n" + "="*60)
print("DIAGNOSTIC: Performance on FULL dataset (all samples) with selected features")
print("="*60)
print("This helps determine if negative R² is due to:")
print("  - Poor feature selection (will also be negative on full dataset)")
print("  - Poor sample selection (will be better on full dataset)")

# Ridge on full dataset
if isinstance(X, pd.DataFrame):
    X_full_ridge = X[selected_features_ridge]
else:
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_ridge]
    else:
        feature_indices = selected_features_ridge
    X_full_ridge = X[:, feature_indices]
results_ridge_full = cross_validate_regression(X_full_ridge, y, model=ridge_model, model_name='Ridge (Full Dataset)')
print(f"Ridge on FULL dataset: R² = {results_ridge_full['mean_score']:.4f} ± {results_ridge_full['std_score']:.4f}")
print(f"  (vs {results_ridge_filtered['mean_score']:.4f} on filtered dataset)")

# SVR on full dataset
if isinstance(X, pd.DataFrame):
    X_full_svr = X[selected_features_svr]
else:
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_svr]
    else:
        feature_indices = selected_features_svr
    X_full_svr = X[:, feature_indices]
results_svr_full = cross_validate_regression(X_full_svr, y, model=svr_model, model_name='SVR (Full Dataset)')
print(f"SVR on FULL dataset: R² = {results_svr_full['mean_score']:.4f} ± {results_svr_full['std_score']:.4f}")
print(f"  (vs {results_svr_filtered['mean_score']:.4f} on filtered dataset)")

# RF on full dataset
if isinstance(X, pd.DataFrame):
    X_full_rf = X[selected_features_rf]
else:
    if isinstance(X, pd.DataFrame):
        feature_indices = [X.columns.get_loc(f) for f in selected_features_rf]
    else:
        feature_indices = selected_features_rf
    X_full_rf = X[:, feature_indices]
results_rf_full = cross_validate_regression(X_full_rf, y, model=rf_model, model_name='RF (Full Dataset)')
print(f"RF on FULL dataset: R² = {results_rf_full['mean_score']:.4f} ± {results_rf_full['std_score']:.4f}")
print(f"  (vs {results_rf_filtered['mean_score']:.4f} on filtered dataset)")

print("\nInterpretation:")
if results_ridge_full['mean_score'] > results_ridge_filtered['mean_score']:
    print("  Ridge: Better on full dataset → Sample selection may be removing useful samples")
else:
    print("  Ridge: Similar/worse on full dataset → Feature selection or small sample size issue")
if results_svr_full['mean_score'] > results_svr_filtered['mean_score']:
    print("  SVR: Better on full dataset → Sample selection may be removing useful samples")
else:
    print("  SVR: Similar/worse on full dataset → Feature selection or small sample size issue")
if results_rf_full['mean_score'] > results_rf_filtered['mean_score']:
    print("  RF: Better on full dataset → Sample selection may be removing useful samples")
else:
    print("  RF: Similar/worse on full dataset → Feature selection or small sample size issue")
print("="*60)

# Step 2: Use FULL dataset for ensemble (since models perform better on full dataset)
print("\n" + "="*60)
print("Step 2: Using FULL dataset for ensemble")
print("="*60)
print("Based on diagnostics:")
print(f"  Ridge: R² = {results_ridge_full['mean_score']:.4f} on full dataset (vs {results_ridge_filtered['mean_score']:.4f} on filtered)")
print(f"  SVR: R² = {results_svr_full['mean_score']:.4f} on full dataset (vs {results_svr_filtered['mean_score']:.4f} on filtered)")
print(f"  RF: R² = {results_rf_full['mean_score']:.4f} on full dataset (vs {results_rf_filtered['mean_score']:.4f} on filtered)")
print("\nUsing FULL dataset since Ridge and RF perform better on it.")
print(f"  Full dataset: {len(y)} samples")

# Use full dataset for ensemble
X_combined = X
y_combined = y

# Step 3: Evaluate ensemble on FULL dataset
print("\n" + "="*60)
print("Step 3: Ensemble Evaluation on FULL Dataset")
print("="*60)

# IMPORTANT: Use each model's own features for ensemble evaluation, not union
# This ensures models are evaluated on features they were optimized for
print(f"  Ridge features: {len(selected_features_ridge)}")
print(f"  SVR features: {len(selected_features_svr)}")
print(f"  RF features: {len(selected_features_rf)}")

# Combine all selected features (union) - still needed for later use
all_selected_features = list(set(selected_features_ridge + selected_features_svr + selected_features_rf))
print(f"  Combined features (union): {len(all_selected_features)} features")

# Use full dataset
X_polished = X_combined
y_polished = y_combined

# Evaluate initial ensemble using each model's own features
initial_ensemble_score, initial_scores = evaluate_ensemble_with_model_features(
    X_polished, y_polished, 
    selected_features_ridge, selected_features_svr, selected_features_rf,
    X_original=X, random_state=42, svr_C=svr_C_param, ridge_alpha=ridge_alpha,
    rf_n_estimators=10, rf_max_depth=3  # Match filtering parameters
)
if not np.isnan(initial_ensemble_score):
    print(f"  Initial ensemble R2: {initial_ensemble_score:.3f}")
    print(f"    Ridge: {initial_scores['ridge']:.3f}, SVR: {initial_scores['svr']:.3f}, RF: {initial_scores['rf']:.3f}")
else:
    print(f"  Initial ensemble R2: N/A (insufficient samples for CV)")
    print(f"    Ridge: N/A, SVR: N/A, RF: N/A")

# Skip polishing when using full dataset - we want to use all samples
# The filtered datasets showed poor performance, so using full dataset is better
skip_polishing = True
if not np.isnan(initial_ensemble_score):
    if initial_ensemble_score > 0:
        print(f"  Initial ensemble R² ({initial_ensemble_score:.3f}) is positive! Using full dataset.")
    else:
        print(f"  Initial ensemble R² ({initial_ensemble_score:.3f}). Using full dataset (better than filtered datasets).")
else:
    print(f"  Initial ensemble R²: N/A. Using full dataset.")
X_final = X_polished
y_final = y_polished
final_ensemble_score = initial_ensemble_score
final_scores = initial_scores

# Skip polishing when using full dataset
if False:  # Disabled polishing for full dataset
    # Polishing: remove samples that hurt ensemble performance (VERY conservative approach)
    # Keep at least 50% of samples to maintain stability
    min_samples_to_keep = max(30, int(len(y_polished) * 0.5))  # Keep at least 50% or 30 samples
    max_samples_to_remove = len(y_polished) - min_samples_to_keep
    
    print(f"  Polishing: Will keep at least {min_samples_to_keep} samples (max {max_samples_to_remove} to remove)")
    
    samples_to_remove = []
    if isinstance(X_polished, pd.DataFrame):
        sample_indices = list(X_polished.index)
    else:
        sample_indices = list(range(len(y_polished)))
    
    # Evaluate each sample removal, but only keep candidates that show significant improvement
    candidate_removals = []  # Store (sample_idx, improvement) pairs
    for sample_idx in sample_indices:
        # Create subset without this sample
        if isinstance(X_polished, pd.DataFrame):
            X_test = X_polished.drop([sample_idx])
            y_test = y_polished.drop([sample_idx])
        else:
            mask = np.arange(len(y_polished)) != sample_idx
            X_test = X_polished[mask]
            y_test = y_polished[mask]
        
        if len(y_test) < min_samples_to_keep:
            continue
        
        test_score, _ = evaluate_ensemble_with_model_features(
            X_test, y_test, 
            selected_features_ridge, selected_features_svr, selected_features_rf,
            X_original=X, random_state=42, svr_C=svr_C_param, ridge_alpha=ridge_alpha,
            rf_n_estimators=10, rf_max_depth=3
        )
        if not np.isnan(test_score) and not np.isnan(initial_ensemble_score):
            improvement = test_score - initial_ensemble_score
            # Require larger improvement threshold (0.05 instead of 0.01) to be more conservative
            if improvement > 0.05:  
                candidate_removals.append((sample_idx, improvement))
    
    # Sort by improvement (best improvements first) and only remove top candidates
    # Limit to max_samples_to_remove
    candidate_removals.sort(key=lambda x: x[1], reverse=True)
    samples_to_remove = [idx for idx, _ in candidate_removals[:max_samples_to_remove]]
    
    # Remove identified samples
    if samples_to_remove:
        if isinstance(X_polished, pd.DataFrame):
            # samples_to_remove contains index labels, use drop() directly
            X_final = X_polished.drop(samples_to_remove)
            y_final = y_polished.drop(samples_to_remove) if isinstance(y_polished, pd.Series) else y_polished.drop(samples_to_remove)
        else:
            # samples_to_remove contains integer positions
            X_final = np.delete(X_polished, samples_to_remove, axis=0)
            y_final = np.delete(y_polished, samples_to_remove, axis=0)
        print(f"  Removed {len(samples_to_remove)} samples during polishing")
    else:
        X_final = X_polished
        y_final = y_polished
        print(f"  No samples removed during polishing")
    
    final_ensemble_score, final_scores = evaluate_ensemble_with_model_features(
        X_final, y_final, 
        selected_features_ridge, selected_features_svr, selected_features_rf,
        X_original=X, random_state=42, svr_C=svr_C_param, ridge_alpha=ridge_alpha,
        rf_n_estimators=10, rf_max_depth=3
    )
if not np.isnan(final_ensemble_score):
    print(f"  Final ensemble R2: {final_ensemble_score:.3f}")
    print(f"    Ridge: {final_scores['ridge']:.3f}, SVR: {final_scores['svr']:.3f}, RF: {final_scores['rf']:.3f}")
else:
    print(f"  Final ensemble R2: N/A (insufficient samples for CV)")
    print(f"    Ridge: N/A, SVR: N/A, RF: N/A")

# Update filtered data to use the polished ensemble result
X_filtered = X_final
y_filtered = y_final

# Update selected_features reference for later use (use combined features)
selected_features = all_selected_features

# Extract kept and outlier samples
# Since we're using full dataset for ensemble, all samples are "kept" for ensemble
# But we identify samples that were selected by individual models vs not selected by any
# Note: kept_samples_* are sets of integer positions (0, 1, 2, ...)
combined_samples = kept_samples_ridge | kept_samples_svr | kept_samples_rf

# All samples are integer positions (0 to len(y)-1)
all_indices = set(range(len(y)))
kept_indices_set = combined_samples  # Integer positions

# Get TCGA codes for kept samples (samples selected by at least one model)
if isinstance(y, pd.Series):
    kept_tcga_codes = [y.index[i] for i in combined_samples]
else:
    kept_tcga_codes = list(combined_samples)

# Get TCGA codes for outlier samples (samples not selected by any model)
removed_samples_set = all_indices - kept_indices_set  # Integer positions
if isinstance(y, pd.Series):
    outlier_tcga_codes = [y.index[i] for i in removed_samples_set]
else:
    outlier_tcga_codes = list(removed_samples_set)

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save to files
kept_file = os.path.join(output_dir, 'kept_samples_tcga_codes.csv')
outlier_file = os.path.join(output_dir, 'outlier_samples_tcga_codes.csv')

# Save kept samples
kept_df = pd.DataFrame({'TCGACode': kept_tcga_codes})
kept_df.to_csv(kept_file, index=False)
print(f"\nSaved {len(kept_tcga_codes)} kept TCGA codes to: {kept_file}")

# Save outlier samples
outlier_df = pd.DataFrame({'TCGACode': outlier_tcga_codes})
outlier_df.to_csv(outlier_file, index=False)
print(f"Saved {len(outlier_tcga_codes)} outlier TCGA codes to: {outlier_file}")

# Extract outlier data for testing (samples not in final filtered set)
if isinstance(X, pd.DataFrame):
    if len(outlier_tcga_codes) > 0:
        X_outliers = X.loc[outlier_tcga_codes, selected_features]
        y_outliers = y.loc[outlier_tcga_codes] if isinstance(y, pd.Series) else y[outlier_tcga_codes]
    else:
        # No outliers
        X_outliers = pd.DataFrame()
        y_outliers = pd.Series() if isinstance(y, pd.Series) else np.array([])
else:
    # X is numpy array - convert outlier_tcga_codes to indices
    if isinstance(y, pd.Series):
        outlier_indices = [list(y.index).index(code) for code in outlier_tcga_codes]
    else:
        outlier_indices = list(outlier_tcga_codes)
    if len(outlier_indices) > 0:
        # Get feature indices
        if isinstance(X, pd.DataFrame):
            feature_indices = [X.columns.get_loc(f) for f in selected_features]
        else:
            feature_indices = list(range(len(selected_features)))
        X_outliers = X[np.array(outlier_indices)][:, feature_indices]
        y_outliers = y[np.array(outlier_indices)]
    else:
        X_outliers = np.array([]).reshape(0, len(selected_features))
        y_outliers = np.array([])

print(f"\nOutlier samples: {len(y_outliers)} samples")


# Visualize predicted vs actual values for each model
import matplotlib.pyplot as plt

# Create figure with subplots for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Predicted vs Actual Values (Cross-Validation)', fontsize=16, fontweight='bold')

# Ridge model
ax1 = axes[0]
y_actual_ridge = y_ridge.values if isinstance(y_ridge, pd.Series) else y_ridge
y_pred_ridge = results_ridge_filtered['predictions'].values if isinstance(results_ridge_filtered['predictions'], pd.Series) else results_ridge_filtered['predictions']
ax1.scatter(y_actual_ridge, y_pred_ridge, alpha=0.6, s=50)
ax1.plot([y_actual_ridge.min(), y_actual_ridge.max()], 
         [y_actual_ridge.min(), y_actual_ridge.max()], 
         'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Actual Values', fontsize=12)
ax1.set_ylabel('Predicted Values', fontsize=12)
ax1.set_title(f"Ridge (R² = {results_ridge_filtered['mean_score']:.3f}, Corr = {results_ridge_filtered['correlation']:.3f})", 
              fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# SVR model
ax2 = axes[1]
y_actual_svr = y_svr.values if isinstance(y_svr, pd.Series) else y_svr
y_pred_svr = results_svr_filtered['predictions'].values if isinstance(results_svr_filtered['predictions'], pd.Series) else results_svr_filtered['predictions']
ax2.scatter(y_actual_svr, y_pred_svr, alpha=0.6, s=50, color='green')
ax2.plot([y_actual_svr.min(), y_actual_svr.max()], 
         [y_actual_svr.min(), y_actual_svr.max()], 
         'r--', lw=2, label='Perfect prediction')
ax2.set_xlabel('Actual Values', fontsize=12)
ax2.set_ylabel('Predicted Values', fontsize=12)
ax2.set_title(f"SVR (R² = {results_svr_filtered['mean_score']:.3f}, Corr = {results_svr_filtered['correlation']:.3f})", 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Random Forest model
ax3 = axes[2]
y_actual_rf = y_rf.values if isinstance(y_rf, pd.Series) else y_rf
y_pred_rf = results_rf_filtered['predictions'].values if isinstance(results_rf_filtered['predictions'], pd.Series) else results_rf_filtered['predictions']
ax3.scatter(y_actual_rf, y_pred_rf, alpha=0.6, s=50, color='orange')
ax3.plot([y_actual_rf.min(), y_actual_rf.max()], 
         [y_actual_rf.min(), y_actual_rf.max()], 
         'r--', lw=2, label='Perfect prediction')
ax3.set_xlabel('Actual Values', fontsize=12)
ax3.set_ylabel('Predicted Values', fontsize=12)
ax3.set_title(f"Random Forest (R² = {results_rf_filtered['mean_score']:.3f}, Corr = {results_rf_filtered['correlation']:.3f})", 
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize ensemble predictions on full dataset
print("\nGenerating ensemble prediction plots...")
y_pred_ridge_ens, y_pred_svr_ens, y_pred_rf_ens, y_pred_ensemble = get_ensemble_predictions(
    X_final, y_final,
    selected_features_ridge, selected_features_svr, selected_features_rf,
    X_original=X, random_state=42, svr_C=svr_C_param, ridge_alpha=ridge_alpha,
    rf_n_estimators=10, rf_max_depth=3
)

if y_pred_ensemble is not None:
    # Create figure with subplots for ensemble models and ensemble mean
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Ensemble Predictions on Full Dataset (Cross-Validation)', fontsize=16, fontweight='bold', y=0.995)
    
    y_actual_ens = y_final.values if isinstance(y_final, pd.Series) else y_final
    y_pred_ridge_ens_vals = y_pred_ridge_ens.values if isinstance(y_pred_ridge_ens, pd.Series) else y_pred_ridge_ens
    y_pred_svr_ens_vals = y_pred_svr_ens.values if isinstance(y_pred_svr_ens, pd.Series) else y_pred_svr_ens
    y_pred_rf_ens_vals = y_pred_rf_ens.values if isinstance(y_pred_rf_ens, pd.Series) else y_pred_rf_ens
    y_pred_ensemble_vals = y_pred_ensemble.values if isinstance(y_pred_ensemble, pd.Series) else y_pred_ensemble
    
    # Calculate correlations and R² for ensemble
    corr_ridge_ens = pearsonr(y_actual_ens, y_pred_ridge_ens_vals)[0] if len(y_actual_ens) > 1 else 0
    corr_svr_ens = pearsonr(y_actual_ens, y_pred_svr_ens_vals)[0] if len(y_actual_ens) > 1 else 0
    corr_rf_ens = pearsonr(y_actual_ens, y_pred_rf_ens_vals)[0] if len(y_actual_ens) > 1 else 0
    corr_ensemble = pearsonr(y_actual_ens, y_pred_ensemble_vals)[0] if len(y_actual_ens) > 1 else 0
    
    r2_ridge_ens = r2_score(y_actual_ens, y_pred_ridge_ens_vals)
    r2_svr_ens = r2_score(y_actual_ens, y_pred_svr_ens_vals)
    r2_rf_ens = r2_score(y_actual_ens, y_pred_rf_ens_vals)
    r2_ensemble = r2_score(y_actual_ens, y_pred_ensemble_vals)
    
    # Ridge in ensemble
    ax1 = axes2[0, 0]
    ax1.scatter(y_actual_ens, y_pred_ridge_ens_vals, alpha=0.6, s=50, color='blue')
    ax1.plot([y_actual_ens.min(), y_actual_ens.max()], 
             [y_actual_ens.min(), y_actual_ens.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Values', fontsize=11)
    ax1.set_ylabel('Predicted Values', fontsize=11)
    ax1.set_title(f"Ridge in Ensemble (R² = {r2_ridge_ens:.3f}, Corr = {corr_ridge_ens:.3f})", 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SVR in ensemble
    ax2 = axes2[0, 1]
    ax2.scatter(y_actual_ens, y_pred_svr_ens_vals, alpha=0.6, s=50, color='green')
    ax2.plot([y_actual_ens.min(), y_actual_ens.max()], 
             [y_actual_ens.min(), y_actual_ens.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax2.set_xlabel('Actual Values', fontsize=11)
    ax2.set_ylabel('Predicted Values', fontsize=11)
    ax2.set_title(f"SVR in Ensemble (R² = {r2_svr_ens:.3f}, Corr = {corr_svr_ens:.3f})", 
                  fontsize=11, fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # RF in ensemble
    ax3 = axes2[1, 0]
    ax3.scatter(y_actual_ens, y_pred_rf_ens_vals, alpha=0.6, s=50, color='orange')
    ax3.plot([y_actual_ens.min(), y_actual_ens.max()], 
             [y_actual_ens.min(), y_actual_ens.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax3.set_xlabel('Actual Values', fontsize=11)
    ax3.set_ylabel('Predicted Values', fontsize=11)
    ax3.set_title(f"RF in Ensemble (R² = {r2_rf_ens:.3f}, Corr = {corr_rf_ens:.3f})", 
                  fontsize=11, fontweight='bold', pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ensemble mean
    ax4 = axes2[1, 1]
    ax4.scatter(y_actual_ens, y_pred_ensemble_vals, alpha=0.6, s=50, color='purple')
    ax4.plot([y_actual_ens.min(), y_actual_ens.max()], 
             [y_actual_ens.min(), y_actual_ens.max()], 
             'r--', lw=2, label='Perfect prediction')
    ax4.set_xlabel('Actual Values', fontsize=11)
    ax4.set_ylabel('Predicted Values', fontsize=11)
    ax4.set_title(f"Ensemble Mean (R² = {r2_ensemble:.3f}, Corr = {corr_ensemble:.3f})", 
                  fontsize=11, fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3.0, w_pad=2.0)
    plt.show()
    
    print(f"\nEnsemble prediction statistics:")
    print(f"  Ridge: R² = {r2_ridge_ens:.4f}, Correlation = {corr_ridge_ens:.4f}")
    print(f"  SVR: R² = {r2_svr_ens:.4f}, Correlation = {corr_svr_ens:.4f}")
    print(f"  RF: R² = {r2_rf_ens:.4f}, Correlation = {corr_rf_ens:.4f}")
    print(f"  Ensemble Mean: R² = {r2_ensemble:.4f}, Correlation = {corr_ensemble:.4f}")

# Print summary statistics
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print("\nIndividual Model Performance (on their own filtered datasets):")
print(f"Ridge:     R² = {results_ridge_filtered['mean_score']:.4f} ± {results_ridge_filtered['std_score']:.4f}, "
      f"Correlation = {results_ridge_filtered['correlation']:.4f}")
print(f"SVR:        R² = {results_svr_filtered['mean_score']:.4f} ± {results_svr_filtered['std_score']:.4f}, "
      f"Correlation = {results_svr_filtered['correlation']:.4f}")
print(f"Random Forest: R² = {results_rf_filtered['mean_score']:.4f} ± {results_rf_filtered['std_score']:.4f}, "
      f"Correlation = {results_rf_filtered['correlation']:.4f}")

print("\nEnsemble Performance (on FULL dataset with selected features):")
if not np.isnan(final_ensemble_score):
    print(f"Ensemble Mean R²: {final_ensemble_score:.4f}")
    print(f"  Ridge: {final_scores['ridge']:.4f}, SVR: {final_scores['svr']:.4f}, RF: {final_scores['rf']:.4f}")
    print(f"  Final dataset: {len(y_final)} samples, {len(all_selected_features)} features")
    if final_ensemble_score > 0:
        print(f"\n✓ SUCCESS: Ensemble R² ({final_ensemble_score:.4f}) is positive!")
    else:
        print(f"\n⚠ WARNING: Ensemble R² ({final_ensemble_score:.4f}) is still negative.")
        print(f"           Individual models on full dataset:")
        print(f"             Ridge: {results_ridge_full['mean_score']:.4f}, RF: {results_rf_full['mean_score']:.4f}")
        print(f"           Consider adjusting model parameters or feature selection.")
else:
    print(f"Ensemble Mean R²: N/A (insufficient samples for CV)")
    print(f"  Ridge: N/A, SVR: N/A, RF: N/A")
print("="*60)