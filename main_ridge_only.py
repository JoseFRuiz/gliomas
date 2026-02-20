# Ridge Classifier Performance Evaluation
# This script focuses solely on the Ridge model performance

from utils import load_data, cross_validate_regression, filter_samples_for_model_with_features
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
gene_tpm_path = os.path.join('data', 'TCGAGliomas_RNAm_Filtrado_QC_verif.csv')

X, y = load_data(gene_tpm_path=gene_tpm_path)

# Set up parameters
n_samples = X.shape[0]
max_outliers = int(n_samples * 0.5)
n_features = 1000  # Number of features to select
ridge_alpha = 100.0  # Regularization strength
band_percentile = 80  # Percentile threshold for prediction band filtering (keep samples within this percentile of error)

print("="*60)
print("RIDGE CLASSIFIER PERFORMANCE EVALUATION")
print("="*60)
print(f"Total samples: {n_samples}")
print(f"Max outliers to remove: {max_outliers}")
print(f"Number of features to select: {n_features}")
print(f"Ridge alpha (regularization): {ridge_alpha}")
print(f"Band filtering percentile: {band_percentile}%")
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

# Initialize variables for band filtering
X_band_filtered = None
y_band_filtered = None
results_ridge_band = None
kept_indices_band = None
removed_indices_band = None

if len(y_values) >= 10:
    n_splits = min(5, max(3, len(y_values) // 3))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    ridge_model_cv = Ridge(alpha=ridge_alpha)
    y_pred_ridge = cross_val_predict(ridge_model_cv, X_full_ridge_values, y_values, cv=cv)
    
    # Preserve index if y is a Series
    if isinstance(y, pd.Series):
        y_pred_ridge = pd.Series(y_pred_ridge, index=y.index)
    
    # Calculate metrics
    r2_ridge = r2_score(y_values, y_pred_ridge)
    corr_ridge = pearsonr(y_values, y_pred_ridge)[0] if len(y_values) > 1 else 0
    
    print(f"  R²: {r2_ridge:.4f}")
    print(f"  Correlation: {corr_ridge:.4f}")
    
    # Step 5: Visualize predictions
    print("\n--- Step 5: Creating visualization ---")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    y_pred_vals = y_pred_ridge.values if isinstance(y_pred_ridge, pd.Series) else y_pred_ridge
    
    ax.scatter(y_values, y_pred_vals, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.5)
    ax.plot([y_values.min(), y_values.max()], 
            [y_values.min(), y_values.max()], 
            'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Values (Survival Days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values (Survival Days)', fontsize=12, fontweight='bold')
    ax.set_title(f'Ridge Classifier Performance\n(R² = {r2_ridge:.4f}, Correlation = {corr_ridge:.4f})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ridge_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved plot to: ridge_performance.png")
    plt.show()
    
    # Step 5b: Filter samples based on distance from perfect prediction line
    print("\n--- Step 5b: Filtering samples within prediction band ---")
    
    # Calculate perpendicular distance from each point to the line y = x
    # Distance = |predicted - actual| / sqrt(2) for line y = x
    # We'll use absolute error as a simpler metric
    abs_errors = np.abs(y_pred_vals - y_values)
    
    # Define threshold: keep samples within a certain percentile or standard deviation
    # Option 1: Use percentile (e.g., keep 80% of samples closest to the line)
    error_threshold_percentile = np.percentile(abs_errors, band_percentile)
    
    # Option 2: Use standard deviation (e.g., keep samples within 2 std dev)
    error_threshold_std = np.mean(abs_errors) + 2 * np.std(abs_errors)
    
    # Use the more restrictive threshold (keeps more samples)
    error_threshold = min(error_threshold_percentile, error_threshold_std)
    
    # Create mask for samples within the band
    # The band is defined by two parallel lines: y = x ± threshold
    # A point (a, p) is within the band if |p - a| <= threshold
    within_band_mask = abs_errors <= error_threshold
    
    # Get indices of kept and removed samples
    if isinstance(y, pd.Series):
        kept_indices_band = y.index[within_band_mask]
        removed_indices_band = y.index[~within_band_mask]
    else:
        kept_indices_band = np.where(within_band_mask)[0]
        removed_indices_band = np.where(~within_band_mask)[0]
    
    # Filter data
    if isinstance(X_full_ridge, pd.DataFrame):
        X_band_filtered = X_full_ridge.loc[kept_indices_band]
    else:
        X_band_filtered = X_full_ridge_values[kept_indices_band]
    
    if isinstance(y, pd.Series):
        y_band_filtered = y.loc[kept_indices_band]
        y_band_values = y_band_filtered.values
    else:
        y_band_filtered = y[kept_indices_band]
        y_band_values = y_band_filtered
    
    y_pred_band_filtered = y_pred_vals[within_band_mask]
    
    # Calculate metrics on filtered subset
    r2_band = r2_score(y_band_values, y_pred_band_filtered)
    corr_band = pearsonr(y_band_values, y_pred_band_filtered)[0] if len(y_band_values) > 1 else 0
    
    print(f"  Error threshold: {error_threshold:.2f} days")
    print(f"  Samples kept: {len(kept_indices_band)} ({100*len(kept_indices_band)/len(y_values):.1f}%)")
    print(f"  Samples removed: {len(removed_indices_band)} ({100*len(removed_indices_band)/len(y_values):.1f}%)")
    print(f"  R² (filtered): {r2_band:.4f} (original: {r2_ridge:.4f})")
    print(f"  Correlation (filtered): {corr_band:.4f} (original: {corr_ridge:.4f})")
    
    # Re-evaluate model on filtered subset with cross-validation
    print("\n  Re-evaluating Ridge on band-filtered subset with cross-validation...")
    results_ridge_band = cross_validate_regression(X_band_filtered, y_band_filtered, 
                                                    model=ridge_model, model_name='Ridge (Band-Filtered)')
    print(f"  R² (CV): {results_ridge_band['mean_score']:.4f} ± {results_ridge_band['std_score']:.4f}")
    print(f"  Correlation (CV): {results_ridge_band['correlation']:.4f}")
    
    # Get NEW cross-validation predictions from filtered dataset
    y_pred_band_cv = results_ridge_band['predictions']
    if isinstance(y_pred_band_cv, pd.Series):
        y_pred_band_cv_values = y_pred_band_cv.values
    else:
        y_pred_band_cv_values = y_pred_band_cv
    
    # Calculate metrics on NEW CV predictions
    r2_band_cv = r2_score(y_band_values, y_pred_band_cv_values)
    corr_band_cv = pearsonr(y_band_values, y_pred_band_cv_values)[0] if len(y_band_values) > 1 else 0
    
    print(f"  R² (from CV predictions): {r2_band_cv:.4f}")
    print(f"  Correlation (from CV predictions): {corr_band_cv:.4f}")
    
    # Visualize filtered results with parallel lines
    print("\n  Creating visualization with prediction band...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Original with band lines
    ax1.scatter(y_values, y_pred_vals, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.5, label='All samples')
    ax1.scatter(y_values[~within_band_mask], y_pred_vals[~within_band_mask], 
                alpha=0.8, s=80, color='red', edgecolors='black', linewidth=0.5, label='Removed samples')
    
    # Perfect prediction line
    ax1.plot([y_values.min(), y_values.max()], 
            [y_values.min(), y_values.max()], 
            'r--', lw=2, label='Perfect prediction (y=x)')
    
    # Parallel lines defining the band
    x_range = np.array([y_values.min(), y_values.max()])
    ax1.plot(x_range, x_range + error_threshold, 
            'g--', lw=2, alpha=0.7, label=f'Upper band (y=x+{error_threshold:.1f})')
    ax1.plot(x_range, x_range - error_threshold, 
            'g--', lw=2, alpha=0.7, label=f'Lower band (y=x-{error_threshold:.1f})')
    
    ax1.set_xlabel('Actual Values (Survival Days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Values (Survival Days)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Original Predictions with Filtering Band\n(R² = {r2_ridge:.4f}, Corr = {corr_ridge:.4f})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: NEW cross-validation predictions on filtered dataset
    ax2.scatter(y_band_values, y_pred_band_cv_values, alpha=0.6, s=60, color='green', 
               edgecolors='black', linewidth=0.5, label='Filtered samples (new CV)')
    ax2.plot([y_band_values.min(), y_band_values.max()], 
            [y_band_values.min(), y_band_values.max()], 
            'r--', lw=2, label='Perfect prediction')
    
    ax2.set_xlabel('Actual Values (Survival Days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Values (Survival Days)', fontsize=12, fontweight='bold')
    ax2.set_title(f'New CV Predictions on Filtered Dataset\n(R² = {r2_band_cv:.4f}, Corr = {corr_band_cv:.4f})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ridge_band_filtered_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved band-filtered plot to: ridge_band_filtered_performance.png")

    # Ensure output dir exists for sample lists and classifier outputs
    output_dir_step5 = 'output'
    os.makedirs(output_dir_step5, exist_ok=True)

    # Save Band sample lists (selected vs filtered out)
    band_kept_ids = np.atleast_1d(kept_indices_band).tolist()
    band_removed_ids = np.atleast_1d(removed_indices_band).tolist()
    pd.DataFrame({'Sample_ID': band_kept_ids}).to_csv(os.path.join(output_dir_step5, 'band_kept_samples.csv'), index=False)
    pd.DataFrame({'Sample_ID': band_removed_ids}).to_csv(os.path.join(output_dir_step5, 'band_removed_samples.csv'), index=False)
    print(f"  Saved Band sample lists: {len(band_kept_ids)} kept, {len(band_removed_ids)} removed")

    # --- Kept vs removed classifier (can we tell them apart from input features?) ---
    print("\n--- Step 5c: Classifier (kept vs removed) with cross-validation ---")
    # Binary labels: 1 = band kept, 0 = band removed
    y_binary = pd.Series(1, index=kept_indices_band).reindex(y.index, fill_value=0).fillna(0).astype(int)
    y_binary_values = y_binary.values
    X_clf = X_full_ridge_values  # use selected features only
    feature_names_clf = selected_features_ridge

    pipe_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
    ])
    n_class0 = (y_binary_values == 0).sum()
    n_class1 = (y_binary_values == 1).sum()
    n_splits_actual = min(5, n_class0, n_class1) if n_class0 and n_class1 else 0
    if n_splits_actual >= 2:
        cv_clf = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=42)
        acc_scores = cross_val_score(pipe_clf, X_clf, y_binary_values, cv=cv_clf, scoring='accuracy', n_jobs=-1)
        print(f"  Accuracy (CV): {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
        try:
            auc_scores = cross_val_score(pipe_clf, X_clf, y_binary_values, cv=cv_clf, scoring='roc_auc', n_jobs=-1)
            print(f"  ROC-AUC (CV): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        except Exception:
            print("  ROC-AUC: not computed (e.g. one class in a fold)")
    else:
        print("  Skipping CV (need at least 2 samples per class for stratified folds).")

    # Feature weights: (1) classifier on full dataset, (2) classifier on ridge-filtered dataset
    # (1) Full dataset
    pipe_full = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
    ])
    pipe_full.fit(X_clf, y_binary_values)
    coef_full = pipe_full.named_steps['clf'].coef_.ravel()
    weights_full_df = pd.DataFrame({
        'Feature': feature_names_clf,
        'Coefficient_full_dataset': coef_full,
        'Abs_Coefficient_full_dataset': np.abs(coef_full)
    }).sort_values('Abs_Coefficient_full_dataset', ascending=False)

    # (2) Ridge-filtered dataset (only ridge-kept samples, same binary labels)
    ridge_kept_pos = np.array(sorted(kept_samples_ridge))
    X_ridge_for_clf = X_clf[ridge_kept_pos]
    y_binary_ridge = y_binary_values[ridge_kept_pos]
    n_class0_r = (y_binary_ridge == 0).sum()
    n_class1_r = (y_binary_ridge == 1).sum()
    if n_class0_r >= 1 and n_class1_r >= 1:
        pipe_filtered = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
        ])
        pipe_filtered.fit(X_ridge_for_clf, y_binary_ridge)
        coef_filtered = pipe_filtered.named_steps['clf'].coef_.ravel()
        weights_filtered_df = pd.DataFrame({
            'Feature': feature_names_clf,
            'Coefficient_filtered_dataset': coef_filtered,
            'Abs_Coefficient_filtered_dataset': np.abs(coef_filtered)
        }).sort_values('Abs_Coefficient_filtered_dataset', ascending=False)
    else:
        weights_filtered_df = pd.DataFrame({
            'Feature': feature_names_clf,
            'Coefficient_filtered_dataset': np.nan,
            'Abs_Coefficient_filtered_dataset': np.nan
        })

    # Merge and save feature weights for classification (full + filtered)
    if 'Coefficient_filtered_dataset' in weights_filtered_df.columns and weights_filtered_df['Coefficient_filtered_dataset'].notna().any():
        classification_weights_df = weights_full_df.merge(
            weights_filtered_df[['Feature', 'Coefficient_filtered_dataset', 'Abs_Coefficient_filtered_dataset']],
            on='Feature', how='outer'
        )
    else:
        classification_weights_df = weights_full_df.copy()
        classification_weights_df['Coefficient_filtered_dataset'] = np.nan
        classification_weights_df['Abs_Coefficient_filtered_dataset'] = np.nan
    classification_weights_df = classification_weights_df.sort_values('Abs_Coefficient_full_dataset', ascending=False)
    classification_weights_file = os.path.join(output_dir_step5, 'classifier_kept_vs_removed_feature_weights.csv')
    classification_weights_df.to_csv(classification_weights_file, index=False)
    print(f"  Saved classification feature weights (full + filtered) to: {classification_weights_file}")

else:
    print("  WARNING: Insufficient samples for cross-validation predictions")
    y_pred_ridge = None

# Step 6: Feature Relevance Analysis
print("\n--- Step 6: Feature Relevance Analysis ---")

# Ensure output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save sample lists: Ridge filtering (kept vs removed)
sample_index = y.index if isinstance(y, pd.Series) else np.arange(len(y))
ridge_kept_ids = [sample_index[i] for i in kept_samples_ridge]
ridge_removed_ids = [sample_index[i] for i in removed_ridge]
pd.DataFrame({'Sample_ID': ridge_kept_ids}).to_csv(os.path.join(output_dir, 'ridge_kept_samples.csv'), index=False)
pd.DataFrame({'Sample_ID': ridge_removed_ids}).to_csv(os.path.join(output_dir, 'ridge_removed_samples.csv'), index=False)
print(f"  Saved Ridge sample lists: {len(ridge_kept_ids)} kept, {len(ridge_removed_ids)} removed")

# Train final model on full dataset to get coefficients
ridge_final = Ridge(alpha=ridge_alpha)
ridge_final.fit(X_full_ridge_values, y_values)

# Get coefficients (feature weights)
coefficients = ridge_final.coef_
feature_importance_df = pd.DataFrame({
    'Feature': selected_features_ridge,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

# Calculate correlation between each feature and target
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

# Sort by absolute coefficient (most important features for prediction)
feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)

print(f"  Top 10 features by absolute coefficient:")
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"    {row['Feature']}: Coef={row['Coefficient']:.4f}, Corr={row['Correlation_with_Target']:.4f}")

# Create visualizations
print("\n  Creating feature importance visualizations...")

# Figure 1: Top features by absolute coefficient
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 20 features by absolute coefficient
top_n = min(20, len(feature_importance_df))
top_features_coef = feature_importance_df.head(top_n)

colors_coef = ['red' if x < 0 else 'blue' for x in top_features_coef['Coefficient']]
ax1.barh(range(len(top_features_coef)), top_features_coef['Coefficient'], color=colors_coef, alpha=0.7)
ax1.set_yticks(range(len(top_features_coef)))
ax1.set_yticklabels(top_features_coef['Feature'], fontsize=9)
ax1.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax1.set_title(f'Top {top_n} Features by Ridge Coefficient\n(Red=Negative, Blue=Positive)', 
              fontsize=13, fontweight='bold', pad=10)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# Top 20 features by absolute correlation with target
feature_importance_df_corr = feature_importance_df.sort_values('Abs_Correlation', ascending=False)
top_features_corr = feature_importance_df_corr.head(top_n)

colors_corr = ['red' if x < 0 else 'blue' for x in top_features_corr['Correlation_with_Target']]
ax2.barh(range(len(top_features_corr)), top_features_corr['Correlation_with_Target'], color=colors_corr, alpha=0.7)
ax2.set_yticks(range(len(top_features_corr)))
ax2.set_yticklabels(top_features_corr['Feature'], fontsize=9)
ax2.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold')
ax2.set_title(f'Top {top_n} Features by Correlation with Target\n(Red=Negative, Blue=Positive)', 
              fontsize=13, fontweight='bold', pad=10)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('ridge_feature_importance.png', dpi=300, bbox_inches='tight')
print("  Saved feature importance plot to: ridge_feature_importance.png")
plt.show()

# Figure 2: Scatter plot of Coefficient vs Correlation
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

scatter = ax.scatter(feature_importance_df['Correlation_with_Target'], 
                     feature_importance_df['Coefficient'],
                     c=feature_importance_df['Abs_Coefficient'],
                     s=100, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)

ax.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold')
ax.set_ylabel('Ridge Coefficient', fontsize=12, fontweight='bold')
ax.set_title('Feature Relevance: Coefficient vs Correlation with Target\n(Color = Absolute Coefficient)', 
             fontsize=13, fontweight='bold', pad=15)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Absolute Coefficient', fontsize=11, fontweight='bold')

# Annotate top 5 features
top_5 = feature_importance_df.head(5)
for idx, row in top_5.iterrows():
    ax.annotate(row['Feature'], 
                (row['Correlation_with_Target'], row['Coefficient']),
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('ridge_coefficient_vs_correlation.png', dpi=300, bbox_inches='tight')
print("  Saved coefficient vs correlation plot to: ridge_coefficient_vs_correlation.png")
plt.show()

# Save detailed feature importance to CSV
importance_file = os.path.join(output_dir, 'ridge_feature_importance.csv')
feature_importance_df.to_csv(importance_file, index=False)
print(f"  Saved feature importance data to: {importance_file}")

# Statistical summary
print("\n  Feature Importance Statistics:")
print(f"    Mean absolute coefficient: {feature_importance_df['Abs_Coefficient'].mean():.4f}")
print(f"    Median absolute coefficient: {feature_importance_df['Abs_Coefficient'].median():.4f}")
print(f"    Max absolute coefficient: {feature_importance_df['Abs_Coefficient'].max():.4f}")
print(f"    Min absolute coefficient: {feature_importance_df['Abs_Coefficient'].min():.4f}")
print(f"    Mean absolute correlation: {feature_importance_df['Abs_Correlation'].mean():.4f}")
print(f"    Median absolute correlation: {feature_importance_df['Abs_Correlation'].median():.4f}")

# Step 7: Summary statistics
print("\n" + "="*60)
print("RIDGE CLASSIFIER PERFORMANCE SUMMARY")
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
    
    if results_ridge_band is not None:
        print(f"\nBand-Filtered Dataset ({len(kept_indices_band)} samples, removed {len(removed_indices_band)}):")
        print(f"  R² (CV): {results_ridge_band['mean_score']:.4f} ± {results_ridge_band['std_score']:.4f}")
        print(f"  Correlation (CV): {results_ridge_band['correlation']:.4f}")
        print(f"  Improvement in correlation: {results_ridge_band['correlation'] - corr_ridge:+.4f}")

print(f"\nSelected Features: {len(selected_features_ridge)}")
print(f"Ridge Alpha (Regularization): {ridge_alpha}")
print("="*60)

# Save selected features to file
features_file = os.path.join(output_dir, 'ridge_selected_features.csv')
features_df = pd.DataFrame({'Feature': selected_features_ridge})
features_df.to_csv(features_file, index=False)
print(f"\nSaved selected features to: {features_file}")
