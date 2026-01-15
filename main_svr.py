# Load data
from utils import load_data, cross_validate_regression, filter_samples_for_model_with_features
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
import pandas as pd
import os
import matplotlib.pyplot as plt

gene_tpm_path = os.path.join('data', 'TCGAGliomas_RNAm_Filtrado_QC_verif.csv')

X, y = load_data(gene_tpm_path=gene_tpm_path)

sel_rand_var = False

if sel_rand_var:
    num_var = 554
    rand_indexes = np.random.randint(0, X.shape[1], num_var)
    X = X.iloc[:, rand_indexes]

# Set up parameters for SVR filtering
n_samples = X.shape[0]
max_outliers = int(n_samples * 0.5)
print(f"max_outliers to remove: {max_outliers}")
print("\n" + "="*60)
print("SVR FILTERING AND CROSS-VALIDATION")
print("="*60)

# SVR parameters to use during filtering and evaluation
svr_C_param = 1000.0
svr_gamma_param = 'scale'
svr_epsilon_param = 0.01

# Filter for SVR (starting from beginning)
print("\n--- Filtering for SVR Model (starting from beginning) ---")
# Create base SVR model (parameters will be overridden during filtering via svr_C, svr_gamma, svr_epsilon)
svr_model = SklearnPipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=svr_C_param, gamma=svr_gamma_param, epsilon=svr_epsilon_param))
])

# Use all features (no feature selection)
n_all_features = X.shape[1]
print(f"  Using all {n_all_features} features (no feature selection)")

X_svr, y_svr, kept_samples_svr, removed_svr, selected_features_svr = filter_samples_for_model_with_features(
    X, y, svr_model, n_features=n_all_features, max_outliers_to_remove=max_outliers, min_improvement=0.001, 
    random_state=42, model_name="SVR", svr_C=svr_C_param, svr_gamma=svr_gamma_param, svr_epsilon=svr_epsilon_param
)
print(f"  SVR: {len(kept_samples_svr)} samples selected, {len(selected_features_svr)} features selected")


# Check if filtering stopped early due to N/A initial R²
# The function stops early and returns exactly the initial samples (2*5 = 10) if initial evaluation fails
# If we have exactly the minimum initial samples, filtering likely stopped early
min_initial_samples = 2 * 5  # 2 * n_fold
if len(kept_samples_svr) == min_initial_samples:
    print(f"\nERROR: SVR filtering stopped early - cannot evaluate initial samples (R² = N/A)")
    print(f"The function returned exactly the initial {min_initial_samples} samples, indicating filtering failed.")
    print("Stopping execution.")
    import sys
    sys.exit(1)

# Use the same model for cross-validation (create a fresh instance to ensure clean state)
# The parameters are already set correctly from above
svr_model_for_cv = SklearnPipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=svr_C_param, gamma=svr_gamma_param, epsilon=svr_epsilon_param))
])

results_svr_filtered = cross_validate_regression(X_svr, y_svr, model=svr_model_for_cv, model_name='SVR (Filtered)')

# Check if SVR failed to evaluate initial samples
if np.isnan(results_svr_filtered['mean_score']):
    print(f"\nERROR: SVR cannot evaluate initial samples (R² = N/A)")
    print("Stopping execution.")
    import sys
    sys.exit(1)

print(f"  R2 (CV): {results_svr_filtered['mean_score']:.3f}, Correlation: {results_svr_filtered['correlation']:.3f}")

# Visualize predicted vs actual values for SVR
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle('SVR: Predicted vs Actual Values (Cross-Validation)', fontsize=16, fontweight='bold')

y_actual_svr = y_svr.values if isinstance(y_svr, pd.Series) else y_svr
y_pred_svr = results_svr_filtered['predictions'].values if isinstance(results_svr_filtered['predictions'], pd.Series) else results_svr_filtered['predictions']
ax.scatter(y_actual_svr, y_pred_svr, alpha=0.6, s=50, color='green')
ax.plot([y_actual_svr.min(), y_actual_svr.max()], 
         [y_actual_svr.min(), y_actual_svr.max()], 
         'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Values', fontsize=12)
ax.set_ylabel('Predicted Values', fontsize=12)
ax.set_title(f"SVR (R² = {results_svr_filtered['mean_score']:.3f}, Corr = {results_svr_filtered['correlation']:.3f})", 
              fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SVR MODEL PERFORMANCE SUMMARY (Filtered Data)")
print("="*60)
print(f"SVR:        R² = {results_svr_filtered['mean_score']:.4f} ± {results_svr_filtered['std_score']:.4f}, "
      f"Correlation = {results_svr_filtered['correlation']:.4f}")
print(f"  Samples: {len(kept_samples_svr)}")
print(f"  Features: {len(selected_features_svr)}")
print("="*60)

