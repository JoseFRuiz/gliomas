# Ridge Classifier Performance Evaluation
# This script focuses solely on the Ridge model performance

from utils import load_data, cross_validate_regression, filter_samples_for_model_with_features
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
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
n_features = 100  # Number of features to select
ridge_alpha = 10.0  # Regularization strength

print("="*60)
print("RIDGE CLASSIFIER PERFORMANCE EVALUATION")
print("="*60)
print(f"Total samples: {n_samples}")
print(f"Max outliers to remove: {max_outliers}")
print(f"Number of features to select: {n_features}")
print(f"Ridge alpha (regularization): {ridge_alpha}")
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
    
else:
    print("  WARNING: Insufficient samples for cross-validation predictions")
    y_pred_ridge = None

# Step 6: Feature Relevance Analysis
print("\n--- Step 6: Feature Relevance Analysis ---")

# Ensure output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

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

print(f"\nSelected Features: {len(selected_features_ridge)}")
print(f"Ridge Alpha (Regularization): {ridge_alpha}")
print("="*60)

# Save selected features to file
features_file = os.path.join(output_dir, 'ridge_selected_features.csv')
features_df = pd.DataFrame({'Feature': selected_features_ridge})
features_df.to_csv(features_file, index=False)
print(f"\nSaved selected features to: {features_file}")
