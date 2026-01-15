import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.pipeline import Pipeline


def load_data(clinical_path='data/ClinicaGliomasDic2025verificados.csv', 
              gene_tpm_path='data/TCGAGliomas_RNAm_Filtrado_QC_DEGCol_verif.csv'):
    """
    Load clinical and gene expression data and prepare input/output variables.
    
    Parameters:
    -----------
    clinical_path : str
        Path to the clinical data CSV file
    gene_tpm_path : str
        Path to the gene TPM expression data CSV file
    
    Returns:
    --------
    X : pandas.DataFrame
        Input features (gene expression data with TCGACodes as rows, genes as columns)
    y : pandas.Series
        Output variable (Sobrevida_dias) aligned with X
    """
    # Load the data
    df_clinical = pd.read_csv(clinical_path)
    df_gene_tpm = pd.read_csv(gene_tpm_path)
    
    # Extract output variable (Sobrevida_dias) from clinical data
    # Use TCGACode as index for alignment
    df_clinical_indexed = df_clinical.set_index('TCGACode')
    y = df_clinical_indexed['Sobrevida_dias']
    
    # The gene_tpm data has TCGACodes as columns
    # Check if first column is gene names (index) or if it's already in the right format
    # If first column is gene names, set it as index before transposing
    if df_gene_tpm.columns[0] not in df_clinical['TCGACode'].values:
        # First column is likely gene names, set it as index
        df_gene_tpm = df_gene_tpm.set_index(df_gene_tpm.columns[0])
    
    # Transpose to have TCGACodes as rows (samples) and genes as columns (features)
    X = df_gene_tpm.T
    
    # Get common TCGACodes between clinical and gene expression data
    common_codes = y.index.intersection(X.index)
    
    # Filter to only common samples
    X = X.loc[common_codes]
    y = y.loc[common_codes]
    
    # Remove samples with missing values in the target variable
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    return X, y


def cross_validate_regression(X, y, model=None, model_name=None, cv=5, scoring='r2', random_state=42):
    """
    Apply cross-validation to predict a continuous variable y using features X.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y : pandas.Series or numpy.ndarray
        Output variable (continuous target)
    model : sklearn estimator, optional
        Regression model to use. If None, defaults to Ridge regression
    model_name : str, optional
        Name of the model to use for the results
    cv : int or cross-validation generator, default=5
        Number of folds for cross-validation
    scoring : str or callable, default='r2'
        Scoring metric to use. Common options:
        - 'r2' (default)
        - 'neg_mean_squared_error'
        - 'neg_mean_absolute_error'
        - 'neg_root_mean_squared_error'
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'scores': array of cross-validation scores
        - 'mean_score': mean of cross-validation scores
        - 'std_score': standard deviation of cross-validation scores
        - 'predictions': array of predictions on test data (same dimensions as y)
        - 'model': the fitted model (fitted on full data)
        - 'model_name': name of the model used
        - 'cv': the cross-validation generator used
    """
    # Store original index if y is a pandas Series
    y_index = y.index if isinstance(y, pd.Series) else None
    
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Default to Ridge regression if no model provided
    if model is None:
        model = Ridge(random_state=random_state)
    
    # Create cross-validation generator
    if isinstance(cv, int):
        cv_generator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_generator = cv
    
    # Perform cross-validation to get scores
    cv_scores = cross_val_score(model, X_values, y_values, cv=cv_generator, scoring=scoring, n_jobs=-1)
    
    # Get predictions on test data for each fold
    y_pred = cross_val_predict(model, X_values, y_values, cv=cv_generator, n_jobs=-1)
    
    # Calculate correlation between predictions and actual values
    if len(y_values) > 1 and np.std(y_values) > 0 and np.std(y_pred) > 0:
        correlation = np.corrcoef(y_values, y_pred)[0, 1]
    else:
        correlation = 0.0
    
    # Restore original index if y was a pandas Series
    if y_index is not None:
        y_pred = pd.Series(y_pred, index=y_index)
    
    # Fit model on full data for reference
    model.fit(X_values, y_values)
    
    # Prepare results
    results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'correlation': correlation,
        'predictions': y_pred,
        'model': model,
        'model_name': model_name,
        'cv': cv_generator
    }
    
    return results


def augment_regression_data(X_train, y_train, method='gaussian_noise', 
                           augmentation_factor=2, noise_level=0.1, random_state=42):
    """
    Augment regression training data using various techniques.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features (samples x features)
    y_train : numpy.ndarray
        Training target values
    method : str, default='gaussian_noise'
        Augmentation method. Options:
        - 'gaussian_noise': Add Gaussian noise to features
        - 'mixup': Interpolate between samples (both X and y) using Beta distribution
        - 'linear_interpolation': SMOTE-like linear interpolation between pairs of samples
        - 'feature_noise': Add noise proportional to feature variance
        - 'combined': Use both gaussian_noise and mixup
    augmentation_factor : int, default=2
        How many times to augment (e.g., 2 = double the dataset size)
    noise_level : float, default=0.1
        Noise level (for gaussian_noise: std as fraction of feature std,
                     for feature_noise: std as fraction of feature std)
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    X_augmented : numpy.ndarray
        Augmented features
    y_augmented : numpy.ndarray
        Augmented target values
    """
    np.random.seed(random_state)
    n_samples, n_features = X_train.shape
    
    if augmentation_factor <= 1:
        return X_train, y_train
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    n_new_samples = int(n_samples * (augmentation_factor - 1))
    
    if method == 'combined':
        # Split samples between the two methods
        n_gaussian = n_new_samples // 2
        n_mixup = n_new_samples - n_gaussian
        n_linear = 0
    else:
        n_gaussian = n_new_samples if method == 'gaussian_noise' else 0
        n_mixup = n_new_samples if method == 'mixup' else 0
        n_linear = n_new_samples if method == 'linear_interpolation' else 0
    
    if n_gaussian > 0:
        # Add Gaussian noise proportional to feature standard deviation
        feature_stds = np.std(X_train, axis=0)
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # Avoid division by zero
        
        for _ in range(n_gaussian):
            # Randomly select a sample to augment
            idx = np.random.randint(0, n_samples)
            noise = np.random.normal(0, noise_level * feature_stds, size=n_features)
            X_aug_list.append((X_train[idx] + noise).reshape(1, -1))
            y_aug_list.append(y_train[idx])
    
    if n_mixup > 0:
        # Mixup: interpolate between pairs of samples
        for _ in range(n_mixup):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            # Random interpolation factor
            alpha = np.random.beta(0.2, 0.2)  # Beta distribution for mixup
            # Interpolate features
            X_mix = alpha * X_train[idx1] + (1 - alpha) * X_train[idx2]
            # Interpolate target (for regression, this makes sense)
            y_mix = alpha * y_train[idx1] + (1 - alpha) * y_train[idx2]
            X_aug_list.append(X_mix.reshape(1, -1))
            y_aug_list.append(y_mix)
    
    if n_linear > 0:
        # SMOTE-like linear interpolation: create samples along the line connecting pairs
        # This preserves linear relationships better than mixup
        for _ in range(n_linear):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            # Uniform interpolation factor (more uniform than Beta for SMOTE-like behavior)
            alpha = np.random.uniform(0, 1)
            # Linear interpolation of features
            X_new = alpha * X_train[idx1] + (1 - alpha) * X_train[idx2]
            # Linear interpolation of target (preserves linear relationship)
            y_new = alpha * y_train[idx1] + (1 - alpha) * y_train[idx2]
            X_aug_list.append(X_new.reshape(1, -1))
            y_aug_list.append(y_new)
    
    if method == 'feature_noise':
        # Add noise proportional to each feature's variance
        feature_stds = np.std(X_train, axis=0)
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)
        
        for _ in range(n_new_samples):
            idx = np.random.randint(0, n_samples)
            noise = np.random.normal(0, noise_level * feature_stds, size=n_features)
            X_aug_list.append((X_train[idx] + noise).reshape(1, -1))
            y_aug_list.append(y_train[idx])
    
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.hstack(y_aug_list)
    
    return X_augmented, y_augmented


def cross_validate_regression_with_augmentation(X, y, model=None, model_name=None, 
                                                cv=5, scoring='r2', random_state=42,
                                                augmentation_method='gaussian_noise',
                                                augmentation_factor=2, noise_level=0.1):
    """
    Apply cross-validation with data augmentation on training sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y : pandas.Series or numpy.ndarray
        Output variable (continuous target)
    model : sklearn estimator, optional
        Regression model to use. If None, defaults to Ridge regression
    model_name : str, optional
        Name of the model to use for the results
    cv : int or cross-validation generator, default=5
        Number of folds for cross-validation
    scoring : str or callable, default='r2'
        Scoring metric to use
    random_state : int, default=42
        Random state for reproducibility
    augmentation_method : str, default='gaussian_noise'
        Augmentation method (see augment_regression_data)
    augmentation_factor : int, default=2
        How many times to augment training data
    noise_level : float, default=0.1
        Noise level for augmentation
    
    Returns:
    --------
    results : dict
        Same format as cross_validate_regression
    """
    # Store original index if y is a pandas Series
    y_index = y.index if isinstance(y, pd.Series) else None
    
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Default to Ridge regression if no model provided
    if model is None:
        model = Ridge(random_state=random_state)
    
    # Create cross-validation generator
    if isinstance(cv, int):
        cv_generator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_generator = cv
    
    # Manual cross-validation with augmentation
    cv_scores = []
    y_pred_all = np.zeros_like(y_values)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_generator.split(X_values)):
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        # Augment training data
        X_train_aug, y_train_aug = augment_regression_data(
            X_train, y_train,
            method=augmentation_method,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level,
            random_state=random_state + fold_idx  # Different seed per fold
        )
        
        # Fit model on augmented training data
        # Create a fresh copy of the model for this fold
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            model_fold = type(model)(**model_params)
        else:
            model_fold = model
        
        model_fold.fit(X_train_aug, y_train_aug)
        
        # Predict on test set
        y_pred_test = model_fold.predict(X_test)
        y_pred_all[test_idx] = y_pred_test
        
        # Calculate score
        if scoring == 'r2':
            score = r2_score(y_test, y_pred_test)
        else:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring)
            score = scorer._score_func(y_test, y_pred_test, **scorer._kwargs)
        cv_scores.append(score)
    
    cv_scores = np.array(cv_scores)
    
    # Restore original index if y was a pandas Series
    if y_index is not None:
        y_pred_all = pd.Series(y_pred_all, index=y_index)
    
    # Fit model on full augmented data for reference
    X_full_aug, y_full_aug = augment_regression_data(
        X_values, y_values,
        method=augmentation_method,
        augmentation_factor=augmentation_factor,
        noise_level=noise_level,
        random_state=random_state + 1000
    )
    model.fit(X_full_aug, y_full_aug)
    
    # Prepare results
    results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'predictions': y_pred_all,
        'model': model,
        'model_name': model_name,
        'cv': cv_generator
    }
    
    return results


def _evaluate_ensemble_models(X_subset, y_subset, random_state=42):
    """
    Evaluate a subset of samples with multiple models (Ridge, SVR, Random Forest).
    
    Parameters:
    -----------
    X_subset : numpy.ndarray
        Feature subset (samples x features)
    y_subset : numpy.ndarray
        Target subset
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary with R² scores for each model and ensemble metrics
    """
    results = {}
    
    # Ridge
    try:
        ridge = Ridge(alpha=1.0, random_state=random_state)
        ridge.fit(X_subset, y_subset)
        y_pred_ridge = ridge.predict(X_subset)
        r2_ridge = r2_score(y_subset, y_pred_ridge)
        results['ridge_r2'] = r2_ridge
    except:
        results['ridge_r2'] = -np.inf
    
    # SVR (requires scaling)
    try:
        from sklearn.pipeline import Pipeline as SklearnPipeline
        # Try RBF kernel first, fallback to linear if it fails
        try:
            svr_pipeline = SklearnPipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
            ])
            svr_pipeline.fit(X_subset, y_subset)
            y_pred_svr = svr_pipeline.predict(X_subset)
            r2_svr = r2_score(y_subset, y_pred_svr)
            # Check for NaN or inf
            if np.isnan(r2_svr) or np.isinf(r2_svr):
                raise ValueError("SVR R² is NaN or inf")
            results['svr_r2'] = r2_svr
        except:
            # Fallback to linear kernel
            svr_pipeline = SklearnPipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='linear', C=1.0, epsilon=0.1))
            ])
            svr_pipeline.fit(X_subset, y_subset)
            y_pred_svr = svr_pipeline.predict(X_subset)
            r2_svr = r2_score(y_subset, y_pred_svr)
            if np.isnan(r2_svr) or np.isinf(r2_svr):
                raise ValueError("SVR R² is NaN or inf")
            results['svr_r2'] = r2_svr
    except Exception as e:
        # If SVR completely fails, return -inf
        results['svr_r2'] = -np.inf
    
    # Random Forest
    try:
        rf = RandomForestRegressor(
            n_estimators=50,  # Smaller for faster evaluation
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        )
        rf.fit(X_subset, y_subset)
        y_pred_rf = rf.predict(X_subset)
        r2_rf = r2_score(y_subset, y_pred_rf)
        results['rf_r2'] = r2_rf
    except:
        results['rf_r2'] = -np.inf
    
    # Ensemble metrics
    results['mean_r2'] = np.mean([results['ridge_r2'], results['svr_r2'], results['rf_r2']])
    results['min_r2'] = np.min([results['ridge_r2'], results['svr_r2'], results['rf_r2']])
    results['models_improved'] = sum([r > -np.inf for r in [results['ridge_r2'], results['svr_r2'], results['rf_r2']]])
    
    return results


def filter_data_for_linear_model(X, y, n_features=None, feature_selection_method='f_regression',
                                  remove_outliers=True, outlier_threshold=3.0, 
                                  min_correlation=0.0, max_pvalue=1.0,
                                  min_correlation_when_n_features=None,
                                  iterative_outlier_removal=False,
                                  sample_selection_method='outlier_removal',
                                  min_r2_improvement=0.001,
                                  min_samples_to_keep=None,
                                  max_outliers_to_remove=None, random_state=42):
    """
    Filter data (samples and features) to create a subset suitable for linear regression.
    
    This function:
    1. Selects features that are linearly related to y (via correlation or F-test)
    2. Selects samples using either outlier removal or greedy forward selection
    3. Returns the filtered dataset
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y : pandas.Series or numpy.ndarray
        Output variable (continuous target)
    n_features : int or None, default=None
        Number of top features to select. If None, selects all features that meet
        the min_correlation threshold. If specified, selects top n_features.
    feature_selection_method : str, default='f_regression'
        Method for feature selection. Options:
        - 'f_regression': F-test for regression (default)
        - 'correlation': Pearson correlation with y
    remove_outliers : bool, default=True
        Whether to remove outlier samples based on standardized residuals.
        Only used when sample_selection_method='outlier_removal'.
    outlier_threshold : float, default=3.0
        Threshold (in standard deviations) for identifying outliers.
        Samples with |standardized_residual| > outlier_threshold are removed.
        Lower values (e.g., 2.5, 2.0) are more strict.
    min_correlation : float, default=0.0
        Minimum absolute correlation with y for a feature to be considered.
        Only used when n_features is None. Higher values (e.g., 0.1, 0.2, 0.3) are more strict.
    max_pvalue : float, default=1.0
        Maximum p-value for feature selection (only used with 'f_regression').
        Features with p-value > max_pvalue are excluded. Lower values (e.g., 0.05, 0.01) are more strict.
    min_correlation_when_n_features : float or None, default=None
        When n_features is specified, also enforce this minimum correlation threshold.
        If None, only top N features are selected regardless of correlation.
        Higher values (e.g., 0.1, 0.2) are more strict.
    iterative_outlier_removal : bool, default=False
        If True, iteratively remove outliers: fit model, remove outliers, refit, remove again.
        More strict but may remove many samples.
        Only used when sample_selection_method='outlier_removal'.
    sample_selection_method : str, default='outlier_removal'
        Method for selecting samples. Options:
        - 'outlier_removal': Remove outliers based on residuals (default)
        - 'greedy_forward': Start with 2 best samples, iteratively add samples that improve R2
    min_r2_improvement : float, default=0.001
        Minimum R2 improvement required to add a sample (only used with 'greedy_forward').
        Stops adding samples when no sample improves R2 by at least this amount.
    min_samples_to_keep : int or None, default=None
        [DEPRECATED] Minimum number of samples to keep. Use max_outliers_to_remove instead.
        If both are provided, max_outliers_to_remove takes precedence.
        For 'greedy_forward': continues adding samples until this minimum is reached.
        For 'outlier_removal': stops removing outliers if this minimum would be violated.
    max_outliers_to_remove : int or None, default=None
        Maximum number of outliers/samples to remove. This is the recommended parameter.
        For 'greedy_forward': limits how many samples are excluded (ensures at least
        n_samples - max_outliers_to_remove are kept).
        For 'outlier_removal': stops removing outliers after this many are removed.
        If None, no maximum is enforced.
        Note: If both min_samples_to_keep and max_outliers_to_remove are provided,
        max_outliers_to_remove takes precedence to avoid conflicts.
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'X_filtered': filtered features (pandas.DataFrame or numpy.ndarray, same type as input)
        - 'y_filtered': filtered target variable (pandas.Series or numpy.ndarray, same type as input)
        - 'selected_features': list of selected feature names/indices
        - 'removed_samples': list of removed sample indices/names
        - 'outlier_scores': pandas.Series or numpy.ndarray of standardized residual scores for all original samples
            (absolute value of standardized residuals from initial model). Higher values indicate more extreme outliers.
            Same type and index as input y.
        - 'n_features_original': original number of features
        - 'n_features_selected': number of selected features
        - 'n_samples_original': original number of samples
        - 'n_samples_filtered': number of samples after filtering
        - 'initial_r2_train': Training R2 score of initial linear model before filtering (may be overfitted)
        - 'initial_r2_cv': Cross-validated R2 score of initial linear model before filtering
        - 'initial_correlation': Pearson correlation between predicted and actual values for initial model
        - 'filtered_r2_train': Training R2 score of linear model after filtering (may be overfitted)
        - 'filtered_r2_cv': Cross-validated R2 score of linear model after filtering
        - 'filtered_correlation': Pearson correlation between predicted and actual values for filtered model
        - 'initial_r2': Alias for initial_r2_train (for backward compatibility)
        - 'filtered_r2': Alias for filtered_r2_train (for backward compatibility)
    """
    # Store original format and indices
    is_dataframe = isinstance(X, pd.DataFrame)
    y_index = y.index if isinstance(y, pd.Series) else None
    feature_names = X.columns.tolist() if is_dataframe else None
    sample_names = X.index.tolist() if is_dataframe else None
    
    # Convert to numpy arrays for processing
    if is_dataframe:
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    n_samples_original, n_features_original = X_values.shape
    
    # Step 1: Feature selection with stricter filtering
    if feature_selection_method == 'f_regression':
        # Use F-test for regression
        f_scores, p_values = f_regression(X_values, y_values)
        
        # Calculate correlations for threshold checking
        correlations = np.array([np.corrcoef(X_values[:, i], y_values)[0, 1] 
                               for i in range(n_features_original)])
        
        # Apply p-value filter
        pvalue_mask = p_values <= max_pvalue
        
        if n_features is not None:
            # Select top n_features that also meet p-value and correlation criteria
            # First filter by p-value
            valid_indices = np.where(pvalue_mask)[0]
            
            # Apply min_correlation_when_n_features if specified
            if min_correlation_when_n_features is not None:
                correlation_mask = np.abs(correlations[valid_indices]) >= min_correlation_when_n_features
                valid_indices = valid_indices[correlation_mask]
            
            # Select top n_features by F-score from valid features
            if len(valid_indices) > 0:
                valid_f_scores = f_scores[valid_indices]
                top_n = min(n_features, len(valid_indices))
                top_local_indices = np.argsort(valid_f_scores)[-top_n:][::-1]
                top_indices = valid_indices[top_local_indices]
            else:
                # No features meet criteria, use top N anyway
                top_indices = np.argsort(f_scores)[-n_features:][::-1]
        else:
            # Select features meeting all criteria
            # Filter by p-value
            valid_indices = np.where(pvalue_mask)[0]
            # Filter by min_correlation
            correlation_mask = np.abs(correlations[valid_indices]) >= min_correlation
            valid_indices = valid_indices[correlation_mask]
            # Sort by F-score
            top_indices = valid_indices[np.argsort(f_scores[valid_indices])[::-1]]
    
    elif feature_selection_method == 'correlation':
        # Use correlation-based selection
        correlations = np.array([np.corrcoef(X_values[:, i], y_values)[0, 1] 
                               for i in range(n_features_original)])
        abs_correlations = np.abs(correlations)
        
        if n_features is not None:
            # Select top n_features by absolute correlation
            # Apply min_correlation_when_n_features if specified
            if min_correlation_when_n_features is not None:
                valid_indices = np.where(abs_correlations >= min_correlation_when_n_features)[0]
                if len(valid_indices) > 0:
                    top_n = min(n_features, len(valid_indices))
                    top_local_indices = np.argsort(abs_correlations[valid_indices])[-top_n:][::-1]
                    top_indices = valid_indices[top_local_indices]
                else:
                    top_indices = np.argsort(abs_correlations)[-n_features:][::-1]
            else:
                top_indices = np.argsort(abs_correlations)[-n_features:][::-1]
        else:
            # Select all features above min_correlation
            top_indices = np.where(abs_correlations >= min_correlation)[0]
            # Sort by absolute correlation
            top_indices = top_indices[np.argsort(abs_correlations[top_indices])[::-1]]
    
    else:
        raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}. "
                        f"Use 'f_regression' or 'correlation'")
    
    # Select features
    X_selected = X_values[:, top_indices]
    n_features_selected = len(top_indices)
    
    # Get selected feature names or indices
    if feature_names is not None:
        selected_features = [feature_names[i] for i in top_indices]
    else:
        selected_features = top_indices.tolist()
    
    # Step 2: Sample selection
    selected_sample_indices = None  # Initialize for use in DataFrame creation
    if sample_selection_method == 'greedy_forward':
        # Greedy forward selection: start with 3-5 representative samples, add samples that improve R2
        n_samples = len(y_values)
        selected_sample_indices = []
        remaining_indices = list(range(n_samples))
        
        # Step 1: Select 3-5 representative samples based on y distribution
        # This is more robust than starting with just 2 samples
        n_initial = min(5, max(3, n_samples // 3))  # Start with 5 samples, or 1/3 of data, minimum 3
        
        print(f"Selecting {n_initial} representative initial samples...")
        
        # Select samples that are spread across the y distribution
        # Include samples near median and at different percentiles
        if n_samples >= n_initial:
            # Calculate percentiles for y distribution
            percentiles = np.linspace(10, 90, n_initial)  # Avoid extremes (0 and 100)
            initial_indices = []
            
            for p in percentiles:
                target_y = np.percentile(y_values, p)
                # Find sample closest to this percentile
                distances = np.abs(y_values - target_y)
                closest_idx = np.argmin(distances)
                # Avoid duplicates
                if closest_idx not in initial_indices:
                    initial_indices.append(closest_idx)
            
            # If we didn't get enough unique samples, add samples near median
            while len(initial_indices) < n_initial:
                median_y = np.median(y_values)
                distances = np.abs(y_values - median_y)
                # Sort by distance and pick closest not already selected
                sorted_indices = np.argsort(distances)
                for idx in sorted_indices:
                    if idx not in initial_indices:
                        initial_indices.append(idx)
                        break
                if len(initial_indices) >= n_initial:
                    break
            
            # Ensure we have exactly n_initial samples
            initial_indices = initial_indices[:n_initial]
        else:
            # If we have fewer samples than n_initial, use all
            initial_indices = list(range(n_samples))
        
        # Evaluate initial samples with ensemble of models
        X_initial = X_selected[initial_indices, :]
        y_initial = y_values[initial_indices]
        
        if len(np.unique(y_initial)) > 1:  # Need variation in y
            initial_results = _evaluate_ensemble_models(X_initial, y_initial, random_state=random_state)
            best_mean_r2 = initial_results['mean_r2']
            best_ridge_r2 = initial_results['ridge_r2']
            best_svr_r2 = initial_results['svr_r2']
            best_rf_r2 = initial_results['rf_r2']
            # Use Ridge correlation for display (backward compatibility)
            ridge = Ridge(alpha=1.0, random_state=random_state)
            ridge.fit(X_initial, y_initial)
            y_pred_initial = ridge.predict(X_initial)
            if len(y_initial) > 1 and np.std(y_initial) > 0 and np.std(y_pred_initial) > 0:
                best_correlation = np.corrcoef(y_initial, y_pred_initial)[0, 1]
            else:
                best_correlation = 0.0
        else:
            # Fallback: use first n_initial samples
            initial_indices = list(range(min(n_initial, n_samples)))
            X_initial = X_selected[initial_indices, :]
            y_initial = y_values[initial_indices]
            initial_results = _evaluate_ensemble_models(X_initial, y_initial, random_state=random_state)
            best_mean_r2 = initial_results['mean_r2']
            best_ridge_r2 = initial_results['ridge_r2']
            best_svr_r2 = initial_results['svr_r2']
            best_rf_r2 = initial_results['rf_r2']
            best_correlation = 0.0
        
        selected_sample_indices = initial_indices.copy()
        remaining_indices = [i for i in remaining_indices if i not in selected_sample_indices]
        
        print(f"Initial {len(selected_sample_indices)} samples - Ensemble Mean R²: {best_mean_r2:.4f}")
        print(f"  Ridge R²: {best_ridge_r2:.4f}, SVR R²: {best_svr_r2:.4f}, RF R²: {best_rf_r2:.4f}")
        print(f"Starting with samples: {selected_sample_indices}")
        
        # Step 2: Iteratively add samples that improve ensemble performance
        current_mean_r2 = best_mean_r2
        current_ridge_r2 = best_ridge_r2
        current_svr_r2 = best_svr_r2
        current_rf_r2 = best_rf_r2
        current_correlation = best_correlation
        improvement_count = 0
        max_no_improvement = 10  # Stop after 10 consecutive samples with no improvement
        
        # Calculate minimum samples to keep
        # max_outliers_to_remove takes precedence if both are provided
        if max_outliers_to_remove is not None:
            min_to_keep = max(len(selected_sample_indices), n_samples_original - max_outliers_to_remove)
        elif min_samples_to_keep is not None:
            min_to_keep = min_samples_to_keep
        else:
            min_to_keep = len(selected_sample_indices)  # At least the initial representative samples
        
        # Validate consistency if both are provided
        if min_samples_to_keep is not None and max_outliers_to_remove is not None:
            implied_min = n_samples_original - max_outliers_to_remove
            if min_samples_to_keep > implied_min:
                print(f"Warning: min_samples_to_keep={min_samples_to_keep} conflicts with "
                      f"max_outliers_to_remove={max_outliers_to_remove} (would require keeping at least {implied_min} samples). "
                      f"Using max_outliers_to_remove as the constraint.")
        
        print(f"Target: keep at least {min_to_keep} samples")
        
        # Continue loop until we meet minimum OR run out of improvements/indices
        while len(remaining_indices) > 0 and (improvement_count < max_no_improvement or len(selected_sample_indices) < min_to_keep):
            # Check if we need to add more samples to meet minimum
            if len(selected_sample_indices) < min_to_keep:
                # Force add samples until minimum is met (even if ensemble doesn't improve much)
                # Find the best remaining sample using ensemble evaluation
                best_candidate = None
                best_candidate_mean_r2 = current_mean_r2
                best_candidate_improvements = 0
                
                for candidate_idx in remaining_indices:
                    test_indices = selected_sample_indices + [candidate_idx]
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    
                    # Evaluate with ensemble
                    test_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                    test_mean_r2 = test_results['mean_r2']
                    
                    # Count how many models improve
                    improvements = 0
                    if test_results['ridge_r2'] > current_ridge_r2:
                        improvements += 1
                    if test_results['svr_r2'] > current_svr_r2:
                        improvements += 1
                    if test_results['rf_r2'] > current_rf_r2:
                        improvements += 1
                    
                    # Prefer samples that improve mean R² or improve at least 2 models
                    if test_mean_r2 > best_candidate_mean_r2 or (improvements >= 2 and test_mean_r2 >= best_candidate_mean_r2 - 0.01):
                        best_candidate_mean_r2 = test_mean_r2
                        best_candidate_improvements = improvements
                        best_candidate = candidate_idx
                
                # Always add a sample if we're below minimum, even if ensemble doesn't improve
                if best_candidate is None and len(remaining_indices) > 0:
                    # Take the first remaining sample if no improvement found
                    best_candidate = remaining_indices[0]
                    test_indices = selected_sample_indices + [best_candidate]
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    test_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                    best_candidate_mean_r2 = test_results['mean_r2']
                    best_candidate_improvements = 0
                
                if best_candidate is not None:
                    selected_sample_indices.append(best_candidate)
                    remaining_indices.remove(best_candidate)
                    # Recalculate ensemble scores for updated set
                    test_indices = selected_sample_indices
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    updated_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                    current_mean_r2 = updated_results['mean_r2']
                    current_ridge_r2 = updated_results['ridge_r2']
                    current_svr_r2 = updated_results['svr_r2']
                    current_rf_r2 = updated_results['rf_r2']
                    # Calculate correlation for display
                    ridge = Ridge(alpha=1.0, random_state=random_state)
                    ridge.fit(X_test, y_test)
                    y_pred = ridge.predict(X_test)
                    if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                        current_correlation = np.corrcoef(y_test, y_pred)[0, 1]
                    else:
                        current_correlation = 0.0
                    improvement = current_mean_r2 - best_mean_r2
                    improvement_count = 0
                    print(f"Added sample {best_candidate} (to meet minimum):")
                    print(f"  Ridge R²: {current_ridge_r2:.4f}")
                    print(f"  SVR R²: {current_svr_r2:.4f}")
                    print(f"  RF R²: {current_rf_r2:.4f}")
                    print(f"  Ensemble Mean R²: {current_mean_r2:.4f}, {best_candidate_improvements} models improved")
                    continue
            
            best_candidate = None
            best_candidate_mean_r2 = current_mean_r2
            best_candidate_improvements = 0
            best_candidate_idx = None
            
            # Try each remaining sample with ensemble evaluation
            for candidate_idx in remaining_indices:
                # Add this candidate to selected samples
                test_indices = selected_sample_indices + [candidate_idx]
                X_test = X_selected[test_indices, :]
                y_test = y_values[test_indices]
                
                # Evaluate with ensemble
                test_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                test_mean_r2 = test_results['mean_r2']
                
                # Count how many models improve
                improvements = 0
                if test_results['ridge_r2'] > current_ridge_r2 + min_r2_improvement:
                    improvements += 1
                if test_results['svr_r2'] > current_svr_r2 + min_r2_improvement:
                    improvements += 1
                if test_results['rf_r2'] > current_rf_r2 + min_r2_improvement:
                    improvements += 1
                
                # Accept if: improves mean R² by threshold OR improves at least 2 models
                if (test_mean_r2 > best_candidate_mean_r2 + min_r2_improvement) or (improvements >= 2 and test_mean_r2 > best_candidate_mean_r2):
                    best_candidate_mean_r2 = test_mean_r2
                    best_candidate_improvements = improvements
                    best_candidate = candidate_idx
                    best_candidate_idx = candidate_idx
            
            # Add the best candidate if it improves ensemble
            if best_candidate is not None:
                previous_mean_r2 = current_mean_r2
                selected_sample_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
                # Recalculate ensemble scores for updated set
                test_indices = selected_sample_indices
                X_test = X_selected[test_indices, :]
                y_test = y_values[test_indices]
                updated_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                current_mean_r2 = updated_results['mean_r2']
                current_ridge_r2 = updated_results['ridge_r2']
                current_svr_r2 = updated_results['svr_r2']
                current_rf_r2 = updated_results['rf_r2']
                # Calculate correlation for display
                ridge = Ridge(alpha=1.0, random_state=random_state)
                ridge.fit(X_test, y_test)
                y_pred = ridge.predict(X_test)
                if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                    current_correlation = np.corrcoef(y_test, y_pred)[0, 1]
                else:
                    current_correlation = 0.0
                improvement = current_mean_r2 - previous_mean_r2
                improvement_count = 0
                print(f"Added sample {best_candidate}:")
                print(f"  Ridge R²: {current_ridge_r2:.4f}")
                print(f"  SVR R²: {current_svr_r2:.4f}")
                print(f"  RF R²: {current_rf_r2:.4f}")
                print(f"  Ensemble Mean R²: {current_mean_r2:.4f}, {best_candidate_improvements} models improved (improvement: {improvement:.4f})")
            else:
                improvement_count += 1
                # Check if we still need more samples to meet minimum
                if len(selected_sample_indices) < min_to_keep:
                    # Force add the next best sample even without improvement
                    if len(remaining_indices) > 0:
                        candidate_idx = remaining_indices[0]
                        test_indices = selected_sample_indices + [candidate_idx]
                        X_test = X_selected[test_indices, :]
                        y_test = y_values[test_indices]
                        
                        updated_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
                        current_mean_r2 = updated_results['mean_r2']
                        current_ridge_r2 = updated_results['ridge_r2']
                        current_svr_r2 = updated_results['svr_r2']
                        current_rf_r2 = updated_results['rf_r2']
                        # Calculate correlation for display
                        ridge = Ridge(alpha=1.0, random_state=random_state)
                        ridge.fit(X_test, y_test)
                        y_pred = ridge.predict(X_test)
                        if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                            current_correlation = np.corrcoef(y_test, y_pred)[0, 1]
                        else:
                            current_correlation = 0.0
                        
                        selected_sample_indices.append(candidate_idx)
                        remaining_indices.remove(candidate_idx)
                        improvement_count = 0
                        print(f"Added sample {candidate_idx} (forced to meet minimum):")
                        print(f"  Ridge R²: {current_ridge_r2:.4f}")
                        print(f"  SVR R²: {current_svr_r2:.4f}")
                        print(f"  RF R²: {current_rf_r2:.4f}")
                        print(f"  Ensemble Mean R²: {current_mean_r2:.4f}")
                elif improvement_count < max_no_improvement:
                    # Try next sample anyway if we haven't hit the limit
                    if len(remaining_indices) > 0:
                        # Remove worst remaining sample and continue
                        remaining_indices.pop(0)
        
        # Ensure we've met the minimum - force add more samples if needed
        while len(selected_sample_indices) < min_to_keep and len(remaining_indices) > 0:
            # Force add samples to meet minimum
            candidate_idx = remaining_indices[0]
            test_indices = selected_sample_indices + [candidate_idx]
            X_test = X_selected[test_indices, :]
            y_test = y_values[test_indices]
            
            updated_results = _evaluate_ensemble_models(X_test, y_test, random_state=random_state)
            current_mean_r2 = updated_results['mean_r2']
            current_ridge_r2 = updated_results['ridge_r2']
            current_svr_r2 = updated_results['svr_r2']
            current_rf_r2 = updated_results['rf_r2']
            # Calculate correlation for display
            ridge = Ridge(alpha=1.0, random_state=random_state)
            ridge.fit(X_test, y_test)
            y_pred = ridge.predict(X_test)
            if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                current_correlation = np.corrcoef(y_test, y_pred)[0, 1]
            else:
                current_correlation = 0.0
            
            selected_sample_indices.append(candidate_idx)
            remaining_indices.remove(candidate_idx)
            print(f"Added sample {candidate_idx} (post-loop, to meet minimum):")
            print(f"  Ridge R²: {current_ridge_r2:.4f}")
            print(f"  SVR R²: {current_svr_r2:.4f}")
            print(f"  RF R²: {current_rf_r2:.4f}")
            print(f"  Ensemble Mean R²: {current_mean_r2:.4f}")
        
        # Final filtered data
        selected_sample_indices = sorted(selected_sample_indices)
        X_filtered = X_selected[selected_sample_indices, :]
        y_filtered = y_values[selected_sample_indices]
        n_outliers = n_samples_original - len(selected_sample_indices)
        
        # Get removed sample names
        removed_indices = [i for i in range(n_samples_original) if i not in selected_sample_indices]
        if sample_names is not None:
            removed_samples = [sample_names[i] for i in removed_indices]
        else:
            removed_samples = removed_indices
        
        # Calculate R2 scores (training and cross-validated)
        initial_model = Ridge(alpha=1.0, random_state=random_state)
        initial_model.fit(X_selected, y_values)
        y_pred_initial = initial_model.predict(X_selected)
        initial_r2_train = r2_score(y_values, y_pred_initial)
        
        # Calculate correlation for initial model
        if len(y_values) > 1 and np.std(y_values) > 0 and np.std(y_pred_initial) > 0:
            initial_correlation = np.corrcoef(y_values, y_pred_initial)[0, 1]
        else:
            initial_correlation = 0.0
        
        # Calculate outlier scores for all original samples (based on initial model)
        residuals_initial = y_values - y_pred_initial
        residual_std_initial = np.std(residuals_initial)
        if residual_std_initial > 0:
            outlier_scores_all = np.abs(residuals_initial / residual_std_initial)
        else:
            outlier_scores_all = np.zeros_like(residuals_initial)
        
        # Cross-validated R2 for initial data
        if len(y_values) >= 5:
            cv_scores_initial = cross_val_score(initial_model, X_selected, y_values, cv=min(5, len(y_values)), scoring='r2', n_jobs=-1)
            initial_r2_cv = np.mean(cv_scores_initial)
        else:
            initial_r2_cv = initial_r2_train
        
        filtered_model = Ridge(alpha=1.0, random_state=random_state)
        filtered_model.fit(X_filtered, y_filtered)
        y_pred_filtered = filtered_model.predict(X_filtered)
        filtered_r2_train = r2_score(y_filtered, y_pred_filtered)
        
        # Calculate correlation for filtered model
        if len(y_filtered) > 1 and np.std(y_filtered) > 0 and np.std(y_pred_filtered) > 0:
            filtered_correlation = np.corrcoef(y_filtered, y_pred_filtered)[0, 1]
        else:
            filtered_correlation = 0.0
        
        # Cross-validated R2 for filtered data
        cv_scores_filtered = None
        if len(y_filtered) >= 5:
            cv_scores_filtered = cross_val_score(filtered_model, X_filtered, y_filtered, cv=min(5, len(y_filtered)), scoring='r2', n_jobs=-1)
            filtered_r2_cv = np.mean(cv_scores_filtered)
        else:
            filtered_r2_cv = filtered_r2_train
        
        # Final ensemble evaluation on filtered data
        final_ensemble = _evaluate_ensemble_models(X_filtered, y_filtered, random_state=random_state)
        
        print(f"\nFinal selection: {len(selected_sample_indices)} samples")
        print(f"  Training R2 (Ridge) = {filtered_r2_train:.4f} (may be overfitted)")
        print(f"  Training Correlation (Ridge) = {filtered_correlation:.4f}")
        if cv_scores_filtered is not None:
            print(f"  Cross-validated R2 (Ridge) = {filtered_r2_cv:.4f} ± {np.std(cv_scores_filtered):.4f}")
        else:
            print(f"  Cross-validated R2 (Ridge) = {filtered_r2_cv:.4f} (too few samples for CV)")
        print(f"  Ensemble Training R²: Mean = {final_ensemble['mean_r2']:.4f}")
        print(f"    Ridge: {final_ensemble['ridge_r2']:.4f}, SVR: {final_ensemble['svr_r2']:.4f}, RF: {final_ensemble['rf_r2']:.4f}")
        
        # Create outlier_mask for later use in DataFrame creation
        outlier_mask = np.zeros(n_samples_original, dtype=bool)
        outlier_mask[selected_sample_indices] = True
        
    elif remove_outliers:
        if iterative_outlier_removal:
            # Iterative outlier removal
            X_current = X_selected
            y_current = y_values
            all_removed_indices = []
            max_iterations = 10
            
            for iteration in range(max_iterations):
                # Fit model
                model = Ridge(alpha=1.0, random_state=random_state)
                model.fit(X_current, y_current)
                y_pred = model.predict(X_current)
                residuals = y_current - y_pred
                
                # Calculate standardized residuals
                residual_std = np.std(residuals)
                if residual_std > 0:
                    standardized_residuals = np.abs(residuals / residual_std)
                else:
                    break
                
                # Identify outliers
                outlier_mask = standardized_residuals <= outlier_threshold
                n_outliers_iter = np.sum(~outlier_mask)
                
                # Apply max_outliers_to_remove limit
                if max_outliers_to_remove is not None:
                    total_removed = len(all_removed_indices) + n_outliers_iter
                    if total_removed > max_outliers_to_remove:
                        # Limit this iteration's removals
                        n_can_remove = max_outliers_to_remove - len(all_removed_indices)
                        if n_can_remove <= 0:
                            break  # Already at limit
                        # Only remove worst n_can_remove outliers
                        outlier_scores = standardized_residuals.copy()
                        outlier_scores[outlier_mask] = -np.inf
                        worst_outlier_indices = np.argsort(outlier_scores)[-n_can_remove:]
                        outlier_mask = np.ones(len(y_current), dtype=bool)
                        outlier_mask[worst_outlier_indices] = False
                        n_outliers_iter = n_can_remove
                
                # Apply min_samples_to_keep limit (only if max_outliers_to_remove not set)
                # max_outliers_to_remove already handled above, so this is a secondary check
                if min_samples_to_keep is not None and max_outliers_to_remove is None:
                    n_would_keep = np.sum(outlier_mask)
                    if n_would_keep < min_samples_to_keep:
                        # Don't remove so many
                        n_can_remove = len(y_current) - min_samples_to_keep
                        if n_can_remove <= 0:
                            break  # Can't remove any more
                        outlier_scores = standardized_residuals.copy()
                        worst_outlier_indices = np.argsort(outlier_scores)[-n_can_remove:]
                        outlier_mask = np.ones(len(y_current), dtype=bool)
                        outlier_mask[worst_outlier_indices] = False
                        n_outliers_iter = n_can_remove
                
                if n_outliers_iter == 0:
                    break
                
                # Track removed indices (need to map back to original)
                removed_local = np.where(~outlier_mask)[0]
                # Map to original indices
                if iteration == 0:
                    current_original_indices = np.arange(len(y_values))
                else:
                    current_original_indices = np.array([i for i in range(len(y_values)) 
                                                       if i not in all_removed_indices])
                
                removed_original = current_original_indices[removed_local]
                all_removed_indices.extend(removed_original.tolist())
                
                # Filter samples
                X_current = X_current[outlier_mask, :]
                y_current = y_current[outlier_mask]
            
            # Final filtered data
            X_filtered = X_current
            y_filtered = y_current
            n_outliers = len(all_removed_indices)
            
            # Get removed sample names
            if sample_names is not None:
                removed_samples = [sample_names[i] for i in all_removed_indices]
            else:
                removed_samples = all_removed_indices
            
            # Calculate R2 scores (training and cross-validated)
            initial_model = Ridge(alpha=1.0, random_state=random_state)
            initial_model.fit(X_selected, y_values)
            y_pred_initial = initial_model.predict(X_selected)
            initial_r2_train = r2_score(y_values, y_pred_initial)
            
            # Calculate correlation for initial model
            if len(y_values) > 1 and np.std(y_values) > 0 and np.std(y_pred_initial) > 0:
                initial_correlation = np.corrcoef(y_values, y_pred_initial)[0, 1]
            else:
                initial_correlation = 0.0
            
            # Calculate outlier scores for all original samples (based on initial model)
            residuals_initial = y_values - y_pred_initial
            residual_std_initial = np.std(residuals_initial)
            if residual_std_initial > 0:
                outlier_scores_all = np.abs(residuals_initial / residual_std_initial)
            else:
                outlier_scores_all = np.zeros_like(residuals_initial)
            
            # Cross-validated R2 for initial data
            if len(y_values) >= 5:
                cv_scores_initial = cross_val_score(initial_model, X_selected, y_values, cv=min(5, len(y_values)), scoring='r2', n_jobs=-1)
                initial_r2_cv = np.mean(cv_scores_initial)
            else:
                initial_r2_cv = initial_r2_train
            
            filtered_model = Ridge(alpha=1.0, random_state=random_state)
            filtered_model.fit(X_filtered, y_filtered)
            y_pred_filtered = filtered_model.predict(X_filtered)
            filtered_r2_train = r2_score(y_filtered, y_pred_filtered)
            
            # Calculate correlation for filtered model
            if len(y_filtered) > 1 and np.std(y_filtered) > 0 and np.std(y_pred_filtered) > 0:
                filtered_correlation = np.corrcoef(y_filtered, y_pred_filtered)[0, 1]
            else:
                filtered_correlation = 0.0
            
            # Cross-validated R2 for filtered data
            if len(y_filtered) >= 5:
                cv_scores_filtered = cross_val_score(filtered_model, X_filtered, y_filtered, cv=min(5, len(y_filtered)), scoring='r2', n_jobs=-1)
                filtered_r2_cv = np.mean(cv_scores_filtered)
            else:
                filtered_r2_cv = filtered_r2_train
            
            # Create outlier_mask for later use in DataFrame creation
            outlier_mask = np.ones(len(y_values), dtype=bool)
            outlier_mask[all_removed_indices] = False
        else:
            # Single-pass outlier removal
            # Fit Ridge regression to avoid issues with multicollinearity
            initial_model = Ridge(alpha=1.0, random_state=random_state)
            initial_model.fit(X_selected, y_values)
            y_pred = initial_model.predict(X_selected)
            residuals = y_values - y_pred
            
            # Calculate standardized residuals
            residual_std = np.std(residuals)
            if residual_std > 0:
                standardized_residuals = np.abs(residuals / residual_std)
            else:
                standardized_residuals = np.zeros_like(residuals)
            
            # Identify outliers
            outlier_mask = standardized_residuals <= outlier_threshold
            n_outliers = np.sum(~outlier_mask)
            
            # Apply max_outliers_to_remove limit
            if max_outliers_to_remove is not None and n_outliers > max_outliers_to_remove:
                # Only remove worst max_outliers_to_remove outliers
                outlier_scores = standardized_residuals.copy()
                outlier_scores[outlier_mask] = -np.inf  # Mark non-outliers as worst
                worst_outlier_indices = np.argsort(outlier_scores)[-max_outliers_to_remove:]
                outlier_mask = np.ones(len(y_values), dtype=bool)
                outlier_mask[worst_outlier_indices] = False
                n_outliers = max_outliers_to_remove
            
            # Apply min_samples_to_keep limit (only if max_outliers_to_remove not set)
            if min_samples_to_keep is not None and max_outliers_to_remove is None:
                n_would_keep = np.sum(outlier_mask)
                if n_would_keep < min_samples_to_keep:
                    # Don't remove so many
                    n_can_remove = len(y_values) - min_samples_to_keep
                    if n_can_remove > 0:
                        outlier_scores = standardized_residuals.copy()
                        outlier_scores[outlier_mask] = -np.inf
                        worst_outlier_indices = np.argsort(outlier_scores)[-n_can_remove:]
                        outlier_mask = np.ones(len(y_values), dtype=bool)
                        outlier_mask[worst_outlier_indices] = False
                        n_outliers = n_can_remove
            
            # Store outlier scores for all samples (before filtering)
            outlier_scores_all = standardized_residuals.copy()
            
            # Filter samples
            X_filtered = X_selected[outlier_mask, :]
            y_filtered = y_values[outlier_mask]
            
            # Get removed sample indices/names
            removed_indices = np.where(~outlier_mask)[0]
            if sample_names is not None:
                removed_samples = [sample_names[i] for i in removed_indices]
            else:
                removed_samples = removed_indices.tolist()
            
            # Calculate R2 scores (training and cross-validated)
            initial_r2_train = r2_score(y_values, y_pred)
            
            # Calculate correlation for initial model
            if len(y_values) > 1 and np.std(y_values) > 0 and np.std(y_pred) > 0:
                initial_correlation = np.corrcoef(y_values, y_pred)[0, 1]
            else:
                initial_correlation = 0.0
            
            # Cross-validated R2 for initial data
            initial_model = Ridge(alpha=1.0, random_state=random_state)
            if len(y_values) >= 5:
                cv_scores_initial = cross_val_score(initial_model, X_selected, y_values, cv=min(5, len(y_values)), scoring='r2', n_jobs=-1)
                initial_r2_cv = np.mean(cv_scores_initial)
            else:
                initial_r2_cv = initial_r2_train
            
            filtered_model = Ridge(alpha=1.0, random_state=random_state)
            filtered_model.fit(X_filtered, y_filtered)
            y_pred_filtered = filtered_model.predict(X_filtered)
            filtered_r2_train = r2_score(y_filtered, y_pred_filtered)
            
            # Calculate correlation for filtered model
            if len(y_filtered) > 1 and np.std(y_filtered) > 0 and np.std(y_pred_filtered) > 0:
                filtered_correlation = np.corrcoef(y_filtered, y_pred_filtered)[0, 1]
            else:
                filtered_correlation = 0.0
            
            # Cross-validated R2 for filtered data
            if len(y_filtered) >= 5:
                cv_scores_filtered = cross_val_score(filtered_model, X_filtered, y_filtered, cv=min(5, len(y_filtered)), scoring='r2', n_jobs=-1)
                filtered_r2_cv = np.mean(cv_scores_filtered)
            else:
                filtered_r2_cv = filtered_r2_train
        
    else:
        # No outlier removal
        X_filtered = X_selected
        y_filtered = y_values
        removed_samples = []
        n_outliers = 0
        
        # Calculate R2 scores (training and cross-validated)
        initial_model = Ridge(alpha=1.0, random_state=random_state)
        initial_model.fit(X_filtered, y_filtered)
        y_pred = initial_model.predict(X_filtered)
        initial_r2_train = r2_score(y_filtered, y_pred)
        filtered_r2_train = initial_r2_train
        
        # Calculate correlation for initial and filtered models (same in this case)
        if len(y_filtered) > 1 and np.std(y_filtered) > 0 and np.std(y_pred) > 0:
            initial_correlation = np.corrcoef(y_filtered, y_pred)[0, 1]
            filtered_correlation = initial_correlation
        else:
            initial_correlation = 0.0
            filtered_correlation = 0.0
        
        # Calculate outlier scores for all samples (based on initial model)
        residuals_initial = y_filtered - y_pred
        residual_std_initial = np.std(residuals_initial)
        if residual_std_initial > 0:
            outlier_scores_all = np.abs(residuals_initial / residual_std_initial)
        else:
            outlier_scores_all = np.zeros_like(residuals_initial)
        
        # Cross-validated R2
        if len(y_filtered) >= 5:
            cv_scores = cross_val_score(initial_model, X_filtered, y_filtered, cv=min(5, len(y_filtered)), scoring='r2', n_jobs=-1)
            initial_r2_cv = np.mean(cv_scores)
            filtered_r2_cv = initial_r2_cv
        else:
            initial_r2_cv = initial_r2_train
            filtered_r2_cv = filtered_r2_train
    
    n_samples_filtered = len(y_filtered)
    
    # Restore original format
    if is_dataframe:
        # Create DataFrame with selected features
        # Get correct indices for filtered samples
        if sample_selection_method == 'greedy_forward':
            # Use the selected_sample_indices from greedy forward selection
            filtered_sample_indices = selected_sample_indices
        elif remove_outliers:
            filtered_sample_indices = np.where(outlier_mask)[0]
        else:
            filtered_sample_indices = np.arange(n_samples_original)
        
        # Create index for DataFrame
        if sample_names is not None:
            df_index = [sample_names[i] for i in filtered_sample_indices]
        else:
            df_index = filtered_sample_indices
        
        if feature_names is not None:
            X_filtered = pd.DataFrame(X_filtered, index=df_index, columns=selected_features)
        else:
            X_filtered = pd.DataFrame(X_filtered, index=df_index)
    else:
        # Keep as numpy array
        pass
    
    if isinstance(y, pd.Series):
        # Create Series with filtered samples
        if sample_selection_method == 'greedy_forward':
            # Use the selected_sample_indices from greedy forward selection
            filtered_indices = selected_sample_indices
        elif remove_outliers:
            filtered_indices = np.where(outlier_mask)[0]
        else:
            filtered_indices = np.arange(len(y_values))
        
        if y_index is not None:
            filtered_y_index = [y_index[i] for i in filtered_indices]
        else:
            filtered_y_index = filtered_indices
        y_filtered = pd.Series(y_filtered, index=filtered_y_index, name=y.name if hasattr(y, 'name') else None)
    else:
        # Keep as numpy array
        pass
    
    # Convert outlier_scores_all to Series if input was Series/DataFrame
    if isinstance(y, pd.Series) and y_index is not None:
        outlier_scores_all = pd.Series(outlier_scores_all, index=y_index, name='outlier_score')
    elif is_dataframe and sample_names is not None:
        outlier_scores_all = pd.Series(outlier_scores_all, index=sample_names, name='outlier_score')
    
    # Prepare results
    results = {
        'X_filtered': X_filtered,
        'y_filtered': y_filtered,
        'selected_features': selected_features,
        'removed_samples': removed_samples,
        'outlier_scores': outlier_scores_all,  # Standardized residual scores for all original samples
        'n_features_original': n_features_original,
        'n_features_selected': n_features_selected,
        'n_samples_original': n_samples_original,
        'n_samples_filtered': n_samples_filtered,
        'n_outliers_removed': n_outliers,
        'initial_r2_train': initial_r2_train,
        'initial_r2_cv': initial_r2_cv,
        'initial_correlation': initial_correlation,
        'filtered_r2_train': filtered_r2_train,
        'filtered_r2_cv': filtered_r2_cv,
        'filtered_correlation': filtered_correlation,
        'initial_r2': initial_r2_train,  # Backward compatibility
        'filtered_r2': filtered_r2_train  # Backward compatibility
    }
    
    return results


def binarize_y(y, threshold=365):
    """
    Binarize a continuous variable y based on a threshold.
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Continuous target variable to binarize
    threshold : float, default=25
        Threshold value for binarization. Values <= threshold become 0,
        values > threshold become 1.
    
    Returns:
    --------
    y_binary : pandas.Series or numpy.ndarray
        Binarized variable with same type and index as input y.
        Values <= threshold are 0, values > threshold are 1.
    """
    # Store original index if y is a pandas Series
    y_index = y.index if isinstance(y, pd.Series) else None
    
    # Convert to numpy array for processing
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Binarize: <= threshold -> 0, > threshold -> 1
    y_binary = (y_values > threshold).astype(int)
    
    # Restore original format
    if y_index is not None:
        y_binary = pd.Series(y_binary, index=y_index, name=y.name if hasattr(y, 'name') else None)
    
    return y_binary


def cross_validate_classification(X, y_binary, model=None, model_name=None, cv=5, scoring='roc_auc', random_state=42):
    """
    Apply cross-validation to predict a binary variable y_binary using features X.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y_binary : pandas.Series or numpy.ndarray
        Binary target variable (0 or 1)
    model : sklearn estimator, optional
        Classification model to use. If None, defaults to LogisticRegression
    model_name : str, optional
        Name of the model to use for the results
    cv : int or cross-validation generator, default=5
        Number of folds for cross-validation
    scoring : str or callable, default='roc_auc'
        Scoring metric to use. Common options:
        - 'roc_auc' (default)
        - 'accuracy'
        - 'f1'
        - 'precision'
        - 'recall'
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'scores': array of cross-validation scores
        - 'mean_score': mean of cross-validation scores
        - 'std_score': standard deviation of cross-validation scores
        - 'predictions': array of predictions on test data (same dimensions as y_binary)
        - 'confusion_matrix': confusion matrix (2x2 array for binary classification)
        - 'model': the fitted model (fitted on full data)
        - 'model_name': name of the model used
        - 'cv': the cross-validation generator used
    """
    # Store original index if y_binary is a pandas Series
    y_index = y_binary.index if isinstance(y_binary, pd.Series) else None
    
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y_binary, pd.Series):
        y_values = y_binary.values
    else:
        y_values = y_binary
    
    # Default to LogisticRegression if no model provided
    if model is None:
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Create cross-validation generator - use StratifiedKFold for classification
    if isinstance(cv, int):
        cv_generator = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_generator = cv
    
    # Perform cross-validation to get scores
    cv_scores = cross_val_score(model, X_values, y_values, cv=cv_generator, scoring=scoring, n_jobs=-1)
    
    # Get predictions on test data for each fold
    y_pred = cross_val_predict(model, X_values, y_values, cv=cv_generator, n_jobs=-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_values, y_pred)
    
    # Restore original index if y_binary was a pandas Series
    if y_index is not None:
        y_pred = pd.Series(y_pred, index=y_index)
    
    # Fit model on full data for reference
    model.fit(X_values, y_values)
    
    # Prepare results
    results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'predictions': y_pred,
        'confusion_matrix': cm,
        'model': model,
        'model_name': model_name,
        'cv': cv_generator
    }
    
    return results


def cross_validate_classification_with_feature_selection(X, y_binary, model=None, model_name=None, cv=5, 
                                                          scoring='roc_auc', random_state=42,
                                                          n_features=1000, selection_method='f_classif'):
    """
    Apply cross-validation with feature selection to predict a binary variable y_binary using features X.
    Feature selection is performed within each CV fold to avoid data leakage.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y_binary : pandas.Series or numpy.ndarray
        Binary target variable (0 or 1)
    model : sklearn estimator, optional
        Classification model to use. If None, defaults to LogisticRegression
    model_name : str, optional
        Name of the model to use for the results
    cv : int or cross-validation generator, default=5
        Number of folds for cross-validation
    scoring : str or callable, default='roc_auc'
        Scoring metric to use. Common options:
        - 'roc_auc' (default)
        - 'accuracy'
        - 'f1'
        - 'precision'
        - 'recall'
    random_state : int, default=42
        Random state for reproducibility
    n_features : int, default=1000
        Number of top features to select
    selection_method : str, default='f_classif'
        Feature selection method. Options:
        - 'f_classif': F-test for classification (default)
        - 'mutual_info': Mutual information
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'scores': array of cross-validation scores
        - 'mean_score': mean of cross-validation scores
        - 'std_score': standard deviation of cross-validation scores
        - 'predictions': array of predictions on test data (same dimensions as y_binary)
        - 'confusion_matrix': confusion matrix (2x2 array for binary classification)
        - 'selected_features': list of feature names/indices selected (from final fit)
        - 'n_features': number of features selected
        - 'model': the fitted pipeline (fitted on full data)
        - 'model_name': name of the model used
        - 'cv': the cross-validation generator used
    """
    # Store original index and column names if y_binary and X are pandas objects
    y_index = y_binary.index if isinstance(y_binary, pd.Series) else None
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    
    # Convert to numpy arrays if pandas objects
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y_binary, pd.Series):
        y_values = y_binary.values
    else:
        y_values = y_binary
    
    # Default to LogisticRegression if no model provided
    if model is None:
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Select feature selection method
    if selection_method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_features)
    elif selection_method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}. Use 'f_classif' or 'mutual_info'")
    
    # Create pipeline: feature selection -> model
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('classifier', model)
    ])
    
    # Create cross-validation generator - use StratifiedKFold for classification
    if isinstance(cv, int):
        cv_generator = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_generator = cv
    
    # Perform cross-validation to get scores
    cv_scores = cross_val_score(pipeline, X_values, y_values, cv=cv_generator, scoring=scoring, n_jobs=-1)
    
    # Get predictions on test data for each fold
    y_pred = cross_val_predict(pipeline, X_values, y_values, cv=cv_generator, n_jobs=-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_values, y_pred)
    
    # Restore original index if y_binary was a pandas Series
    if y_index is not None:
        y_pred = pd.Series(y_pred, index=y_index)
    
    # Fit pipeline on full data to get selected features
    pipeline.fit(X_values, y_values)
    selected_mask = pipeline.named_steps['feature_selection'].get_support()
    
    # Get selected feature names or indices
    if feature_names is not None:
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    else:
        selected_features = np.where(selected_mask)[0].tolist()
    
    # Prepare results
    results = {
        'scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'predictions': y_pred,
        'confusion_matrix': cm,
        'selected_features': selected_features,
        'n_features': n_features,
        'model': pipeline,
        'model_name': model_name,
        'cv': cv_generator
    }
    
    return results


def filter_samples_for_model_with_features(X, y, model, n_features=100, max_outliers_to_remove=None, 
                                           min_improvement=0.001, random_state=42, model_name="Model", 
                                           svr_C=None, svr_gamma=None, svr_epsilon=None):
    """
    Filter samples for a specific model, starting from beginning with feature selection.
    
    This function:
    1. Selects features that are relevant for the model using F-regression
    2. Selects samples using greedy forward selection starting with n_fold representative samples
    3. Returns the filtered dataset and selected features
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input features (samples x features)
    y : pandas.Series or numpy.ndarray
        Output variable (continuous target)
    model : sklearn estimator
        Regression model to use for evaluation
    n_features : int, default=100
        Number of top features to select
    max_outliers_to_remove : int or None, default=None
        Maximum number of samples to exclude (ensures at least n_samples - max_outliers_to_remove are kept)
    min_improvement : float, default=0.001
        Minimum R² improvement required to add a sample
    random_state : int, default=42
        Random state for reproducibility
    model_name : str, default="Model"
        Name of the model for progress printing
    svr_C : float or None, default=None
        C parameter for SVR (only used if model is SVR). If None, uses model's current C value.
    svr_gamma : str or float or None, default=None
        gamma parameter for SVR (only used if model is SVR with RBF kernel). If None, uses model's current gamma.
    svr_epsilon : float or None, default=None
        epsilon parameter for SVR (only used if model is SVR). If None, uses model's current epsilon.
    
    Returns:
    --------
    X_filtered : pandas.DataFrame or numpy.ndarray
        Filtered features (same type as input)
    y_filtered : pandas.Series or numpy.ndarray
        Filtered target variable (same type as input)
    kept_set : set
        Set of kept sample indices/names
    removed : set
        Set of removed sample indices/names
    selected_features : list
        List of selected feature names/indices
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.feature_selection import f_regression
    from sklearn.pipeline import Pipeline
    
    def create_model_copy(model_instance):
        """Create a copy of the model, updating SVR parameters if provided."""
        if isinstance(model_instance, Pipeline):
            # For Pipeline, recreate from steps to avoid parameter issues
            new_steps = []
            svr_step_name = None
            
            for step_name, step_model in model_instance.steps:
                if hasattr(step_model, 'kernel'):  # It's an SVR
                    svr_step_name = step_name
                    # Create new SVR with updated parameters
                    svr_params = step_model.get_params()
                    if svr_C is not None:
                        svr_params['C'] = svr_C
                    if svr_gamma is not None:
                        svr_params['gamma'] = svr_gamma
                    if svr_epsilon is not None:
                        svr_params['epsilon'] = svr_epsilon
                    new_svr = type(step_model)(**svr_params)
                    new_steps.append((step_name, new_svr))
                else:
                    # For other steps (like StandardScaler), just copy them
                    step_params = step_model.get_params()
                    new_step = type(step_model)(**step_params)
                    new_steps.append((step_name, new_step))
            
            # Create new Pipeline with updated steps
            return Pipeline(new_steps)
        elif hasattr(model_instance, 'get_params'):
            model_params = model_instance.get_params()
            
            # For direct SVR model
            if hasattr(model_instance, 'kernel'):
                # Update SVR parameters if provided
                if svr_C is not None:
                    model_params['C'] = svr_C
                if svr_gamma is not None:
                    model_params['gamma'] = svr_gamma
                if svr_epsilon is not None:
                    model_params['epsilon'] = svr_epsilon
            
            return type(model_instance)(**model_params)
        else:
            return model_instance
    
    n_samples = len(y)
    if isinstance(y, pd.Series):
        y_values = y.values
        y_index = y.index
    else:
        y_values = y
        y_index = np.arange(len(y))
    
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        feature_names = X.columns.tolist()
    else:
        X_values = X
        feature_names = None
    
    # Step 1: Feature selection for this model
    print(f"  Selecting {n_features} features for {model_name}...")
    f_scores, p_values = f_regression(X_values, y_values)
    if n_features is not None:
        top_features_idx = np.argsort(f_scores)[-n_features:][::-1]
    else:
        top_features_idx = np.argsort(f_scores)[::-1]
    
    selected_features = [feature_names[i] if feature_names else i for i in top_features_idx]
    X_selected = X_values[:, top_features_idx]
    
    # Step 2: Sample selection using greedy forward selection
    print(f"  Starting greedy forward selection for {model_name}...")
    
    # Start with at least 2*n_fold samples to ensure each CV fold has at least 2 validation samples
    n_fold = 5
    min_initial_samples = min(2 * n_fold, n_samples)
    
    if n_samples >= min_initial_samples:
        # Select samples distributed across the target value range
        # Use quantiles to get representative samples
        sorted_indices = np.argsort(y_values)
        quantile_positions = np.linspace(0, len(sorted_indices) - 1, min_initial_samples, dtype=int)
        initial_indices = sorted_indices[quantile_positions].tolist()
    else:
        # If we have fewer samples than min_initial_samples, use all
        initial_indices = list(range(n_samples))
    
    kept_indices = set(initial_indices)
    
    # Evaluate initial set
    initial_r2 = None
    if len(kept_indices) >= 2 * n_fold:
        kept_list = list(kept_indices)
        X_current = X_selected[kept_list]
        y_current = y_values[kept_list]
        try:
            cv = KFold(n_splits=min(5, len(kept_list)), shuffle=True, random_state=random_state)
            scores = []
            n_folds_processed = 0
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_current)):
                if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
                    if model_name == "SVR":  # Debug info for SVR
                        print(f"    Fold {fold_idx}: Skipped (only {len(val_idx)} validation samples)")
                    continue
                try:
                    model_copy = create_model_copy(model)
                    model_copy.fit(X_current[train_idx], y_current[train_idx])
                    y_pred = model_copy.predict(X_current[val_idx])
                    fold_score = r2_score(y_current[val_idx], y_pred)
                    scores.append(fold_score)
                    n_folds_processed += 1
                    if model_name == "SVR":  # Debug info for SVR
                        print(f"    Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}, R²={fold_score:.4f}")
                except Exception as e:
                    if model_name == "SVR":  # Debug info for SVR
                        print(f"    Fold {fold_idx}: Error during fit/predict - {type(e).__name__}: {str(e)}")
                    continue
            initial_r2 = np.mean(scores) if scores else np.nan
            if not np.isnan(initial_r2):
                print(f"  Initial {len(kept_indices)} samples - {model_name} R²: {initial_r2:.4f} ({n_folds_processed} folds)")
            else:
                if model_name == "SVR":  # Debug info for SVR
                    print(f"  Initial {len(kept_indices)} samples - {model_name} R²: N/A (processed {n_folds_processed} folds, {len(scores)} valid scores)")
                else:
                    print(f"  Initial {len(kept_indices)} samples - {model_name} R²: N/A")
        except Exception as e:
            if model_name == "SVR":  # Debug info for SVR
                print(f"  Initial {len(kept_indices)} samples - {model_name} R²: N/A (Exception: {type(e).__name__}: {str(e)})")
            else:
                print(f"  Initial {len(kept_indices)} samples - {model_name} R²: N/A")
    else:
        print(f"  Initial {len(kept_indices)} samples - {model_name} R²: N/A (too few samples for CV)")
        initial_r2 = None
    
    # If initial R² is N/A, stop the process early
    if initial_r2 is None or np.isnan(initial_r2):
        print(f"  Stopping: Cannot evaluate initial samples (R² = N/A)")
        # Return the initial samples without further selection
        kept_list = sorted(list(kept_indices))
        if isinstance(X, pd.DataFrame):
            X_filtered = X.iloc[kept_list]
            y_filtered = y.iloc[kept_list] if isinstance(y, pd.Series) else pd.Series(y_values[kept_list], index=y_index[kept_list])
        else:
            X_filtered = X_values[kept_list]
            y_filtered = y_values[kept_list] if not isinstance(y, pd.Series) else pd.Series(y_values[kept_list], index=y_index[kept_list])
        
        removed = set(range(n_samples)) - kept_indices
        kept_set = set(kept_list)
        return X_filtered, y_filtered, kept_set, removed, selected_features
    
    remaining = set(range(n_samples)) - kept_indices
    max_remove = min(max_outliers_to_remove, n_samples - len(kept_indices)) if max_outliers_to_remove else n_samples - len(kept_indices)
    
    # Calculate minimum number of samples to keep (based on max_outliers_to_remove)
    min_samples_to_keep = n_samples - max_outliers_to_remove if max_outliers_to_remove else 0
    
    # Normalize initial_r2: convert NaN to None for easier handling
    previous_r2 = initial_r2 if (initial_r2 is not None and not np.isnan(initial_r2)) else None
    
    for iteration in range(max_remove):
        best_candidate = None
        best_score = None
        current_score = None
        
        # Evaluate current set (only if we have enough samples)
        current_score = None
        if len(kept_indices) >= 2 * n_fold:
            kept_list = list(kept_indices)
            X_current = X_selected[kept_list]
            y_current = y_values[kept_list]
            try:
                cv = KFold(n_splits=min(5, len(kept_list)), shuffle=True, random_state=random_state)
                scores = []
                for train_idx, val_idx in cv.split(X_current):
                    if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
                        continue
                    model_copy = create_model_copy(model)
                    model_copy.fit(X_current[train_idx], y_current[train_idx])
                    y_pred = model_copy.predict(X_current[val_idx])
                    scores.append(r2_score(y_current[val_idx], y_pred))
                current_score = np.mean(scores) if scores else None
                # Convert NaN to None for consistency
                if current_score is not None and np.isnan(current_score):
                    current_score = None
            except Exception as e:
                # Silently fail - current_score remains None
                current_score = None
        
        # Check if current_score is valid (needed for determining how many candidates to try)
        current_score_valid = current_score is not None and not np.isnan(current_score)
        
        # Try adding each remaining sample (limit to 50 for speed, but try more if we have no valid score yet)
        max_candidates = 50 if current_score_valid or (previous_r2 is not None) else min(100, len(remaining))
        candidates_to_try = list(remaining)[:max_candidates]
        
        # Track candidates with their scores and statistical properties
        candidate_stats = {}   # candidate -> statistical properties
        
        for candidate in candidates_to_try:
            test_indices = kept_indices | {candidate}
            if len(test_indices) < 2 * n_fold:
                continue
                
            test_list = list(test_indices)
            X_test = X_selected[test_list]
            y_test = y_values[test_list]
            
            # Calculate statistical properties for this candidate (for fallback selection)
            y_candidate = y_values[candidate]
            y_current = y_values[list(kept_indices)]
            y_with_candidate = y_values[test_list]
            
            # Statistical criteria:
            # 1. Coverage: how well does this fill gaps in the distribution?
            # 2. Diversity: how much does this add to the spread?
            # 3. Centrality: how close to median/mean?
            if len(y_current) > 0:
                y_current_sorted = np.sort(y_current)
                y_median = np.median(y_current)
                y_mean = np.mean(y_current)
                
                # Calculate gap-filling score (inverse of distance to nearest existing sample)
                distances_to_existing = np.abs(y_current - y_candidate)
                min_distance = np.min(distances_to_existing)
                gap_score = 1.0 / (1.0 + min_distance)  # Higher = fills a gap better
                
                # Calculate diversity score (how much does this expand the range?)
                current_range = np.max(y_current) - np.min(y_current)
                new_range = np.max(y_with_candidate) - np.min(y_with_candidate)
                diversity_score = new_range - current_range  # Higher = more diverse
                # Normalize diversity score (avoid negative values dominating)
                diversity_score = max(0, diversity_score) / (np.std(y_values) + 1e-6) if np.std(y_values) > 0 else 0
                
                # Calculate centrality score (closer to median is better for balance)
                centrality_score = 1.0 / (1.0 + np.abs(y_candidate - y_median) / (np.std(y_current) + 1e-6))
            else:
                gap_score = 1.0
                diversity_score = 1.0
                centrality_score = 1.0
            
            # Combined statistical score (weighted)
            stat_score = 0.4 * gap_score + 0.4 * diversity_score + 0.2 * centrality_score
            candidate_stats[candidate] = {
                'gap_score': gap_score,
                'diversity_score': diversity_score,
                'centrality_score': centrality_score,
                'combined_score': stat_score,
                'y_value': y_candidate
            }
            
            try:
                scores = []
                cv_test = KFold(n_splits=min(5, len(test_list)), shuffle=True, random_state=random_state)
                for train_idx, val_idx in cv_test.split(X_test):
                    if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
                        continue
                    model_copy = create_model_copy(model)
                    model_copy.fit(X_test[train_idx], y_test[train_idx])
                    y_pred = model_copy.predict(X_test[val_idx])
                    scores.append(r2_score(y_test[val_idx], y_pred))
                test_score = np.mean(scores) if scores else None
                
                # Check if test_score is valid (not None and not NaN)
                test_score_valid = test_score is not None and not np.isnan(test_score)
                best_score_valid = best_score is not None and not np.isnan(best_score)
                
                if test_score_valid and (not best_score_valid or test_score > best_score):
                    best_score = test_score
                    best_candidate = candidate
            except:
                continue
        
        # Add best candidate if it improves
        # Allow adding if: we have a valid best_score AND (no current score OR best improves)
        current_score_valid = current_score is not None and not np.isnan(current_score)
        best_score_valid = best_score is not None and not np.isnan(best_score)
        
        if best_candidate is not None and best_score_valid:
            # Determine if we should add this candidate
            should_add = False
            improvement_reason = ""
            
            if not current_score_valid:
                # No valid current score - accept any valid candidate score
                should_add = True
                improvement_reason = " (no baseline to compare)"
            else:
                # Have valid current score - check if it improves or if we need more samples
                improves = best_score > current_score + min_improvement
                need_more_samples = len(kept_indices) < min_samples_to_keep
                
                if improves:
                    # Prefer samples that improve performance
                    should_add = True
                    improvement_reason = " (improvement)"
                elif need_more_samples:
                    # Accept even without improvement if we haven't reached minimum samples to keep
                    should_add = True
                    improvement_reason = " (reaching minimum samples)"
            
            if should_add:
                kept_indices.add(best_candidate)
                remaining.remove(best_candidate)
                prev_r2_val = previous_r2 if (previous_r2 is not None and not np.isnan(previous_r2)) else 0
                improvement = best_score - prev_r2_val
                print(f"  Added sample {best_candidate}: {model_name} R² = {best_score:.4f} (improvement: {improvement:.4f}){improvement_reason}")
                previous_r2 = best_score
            else:
                # No improvement and we have enough samples - stop searching
                break
        elif len(kept_indices) < min_samples_to_keep and len(candidate_stats) > 0:
            # No valid CV score found, but we need more samples - use statistical criterion
            # Select candidate with best statistical score
            best_stat_candidate = max(candidate_stats.keys(), 
                                     key=lambda c: candidate_stats[c]['combined_score'])
            kept_indices.add(best_stat_candidate)
            remaining.remove(best_stat_candidate)
            stats_info = candidate_stats[best_stat_candidate]
            print(f"  Added sample {best_stat_candidate}: {model_name} (statistical selection: "
                  f"gap={stats_info['gap_score']:.3f}, diversity={stats_info['diversity_score']:.3f}, "
                  f"centrality={stats_info['centrality_score']:.3f}, y={stats_info['y_value']:.2f})")
            # Don't update previous_r2 since we don't have a valid score
        elif best_candidate is None:
            # No valid candidate found - check if we need more samples
            if len(kept_indices) < min_samples_to_keep:
                # Still need more samples, try a few more iterations
                if iteration < 10:  # Give it more tries if we need samples
                    continue
                else:
                    # Can't find valid candidates, but we tried - stop
                    break
            else:
                # Have enough samples and no valid candidates - stop
                break
        else:
            # best_candidate exists but best_score is invalid - shouldn't happen, but break to be safe
            break
    
    # Convert back to original format
    kept_list = sorted(list(kept_indices))
    if isinstance(X, pd.DataFrame):
        X_filtered = X.iloc[kept_list]
        y_filtered = y.iloc[kept_list] if isinstance(y, pd.Series) else pd.Series(y_values[kept_list], index=y_index[kept_list])
    else:
        X_filtered = X_values[kept_list]
        y_filtered = y_values[kept_list] if not isinstance(y, pd.Series) else pd.Series(y_values[kept_list], index=y_index[kept_list])
    
    removed = set(range(n_samples)) - kept_indices
    kept_set = set(kept_list)
    return X_filtered, y_filtered, kept_set, removed, selected_features


def evaluate_ensemble(X_subset, y_subset, all_selected_features, X_original=None, random_state=42, svr_C=100.0):
    """
    Evaluate ensemble of models (Ridge, SVR, Random Forest) on a subset of data.
    
    Parameters:
    -----------
    X_subset : pandas.DataFrame or numpy.ndarray
        Feature subset (samples x features)
    y_subset : pandas.Series or numpy.ndarray
        Target subset
    all_selected_features : list
        List of selected feature names/indices
    X_original : pandas.DataFrame or numpy.ndarray, optional
        Original X dataset (needed if X_subset is numpy array and features are named)
        If None and X_subset is DataFrame, uses X_subset
    random_state : int, default=42
        Random state for reproducibility
    svr_C : float, default=100.0
        C parameter for SVR model
    
    Returns:
    --------
    mean_r2 : float
        Mean R² across all three models
    scores : dict
        Dictionary with individual model R² scores: {'ridge': float, 'svr': float, 'rf': float}
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # Determine how to select features
    if isinstance(X_subset, pd.DataFrame):
        X_subset_selected = X_subset[all_selected_features]
    else:
        # Convert feature names to indices
        if X_original is not None and isinstance(X_original, pd.DataFrame):
            feature_indices = [X_original.columns.get_loc(f) for f in all_selected_features]
        else:
            # Assume features are already indices or use range
            feature_indices = list(range(len(all_selected_features)))
        X_subset_selected = X_subset[:, feature_indices]
    
    X_subset_values = X_subset_selected.values if isinstance(X_subset_selected, pd.DataFrame) else X_subset_selected
    y_subset_values = y_subset.values if isinstance(y_subset, pd.Series) else y_subset
    
    if len(y_subset_values) < 5:
        return np.nan, {'ridge': np.nan, 'svr': np.nan, 'rf': np.nan}
    
    cv = KFold(n_splits=min(5, len(y_subset_values)), shuffle=True, random_state=random_state)
    
    # Ridge
    ridge_model = Ridge()
    ridge_scores = []
    for train_idx, val_idx in cv.split(X_subset_values):
        if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
            continue
        ridge_model.fit(X_subset_values[train_idx], y_subset_values[train_idx])
        y_pred = ridge_model.predict(X_subset_values[val_idx])
        ridge_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    ridge_r2 = np.mean(ridge_scores) if ridge_scores else np.nan
    
    # SVR
    svr_model = SklearnPipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=svr_C, gamma='scale', epsilon=0.01))
    ])
    svr_scores = []
    for train_idx, val_idx in cv.split(X_subset_values):
        if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
            continue
        svr_model.fit(X_subset_values[train_idx], y_subset_values[train_idx])
        y_pred = svr_model.predict(X_subset_values[val_idx])
        svr_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    svr_r2 = np.mean(svr_scores) if svr_scores else np.nan
    
    # Random Forest (using better parameters for final ensemble evaluation)
    rf_model = RandomForestRegressor(
        n_estimators=50,  # Balanced for ensemble evaluation (faster than 100, better than 10)
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    rf_scores = []
    for train_idx, val_idx in cv.split(X_subset_values):
        if len(val_idx) < 2:  # Skip folds with fewer than 2 validation samples
            continue
        rf_model.fit(X_subset_values[train_idx], y_subset_values[train_idx])
        y_pred = rf_model.predict(X_subset_values[val_idx])
        rf_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    rf_r2 = np.mean(rf_scores) if rf_scores else np.nan
    
    # Calculate mean only if at least one model has valid scores
    if not np.isnan(ridge_r2) or not np.isnan(svr_r2) or not np.isnan(rf_r2):
        valid_scores = [s for s in [ridge_r2, svr_r2, rf_r2] if not np.isnan(s)]
        mean_r2 = np.mean(valid_scores) if valid_scores else np.nan
    else:
        mean_r2 = np.nan
    
    return mean_r2, {'ridge': ridge_r2, 'svr': svr_r2, 'rf': rf_r2}


def evaluate_ensemble_with_model_features(X_subset, y_subset, 
                                         ridge_features, svr_features, rf_features,
                                         X_original=None, random_state=42, svr_C=100.0,
                                         ridge_alpha=1.0, rf_n_estimators=10, rf_max_depth=3):
    """
    Evaluate ensemble where each model uses its own selected features.
    This ensures models are evaluated on features they were optimized for.
    
    Parameters:
    -----------
    X_subset : pandas.DataFrame or numpy.ndarray
        Feature subset (samples x features)
    y_subset : pandas.Series or numpy.ndarray
        Target subset
    ridge_features : list
        List of feature names/indices for Ridge model
    svr_features : list
        List of feature names/indices for SVR model
    rf_features : list
        List of feature names/indices for Random Forest model
    X_original : pandas.DataFrame or numpy.ndarray, optional
        Original X dataset (needed if X_subset is numpy array and features are named)
        If None and X_subset is DataFrame, uses X_subset
    random_state : int, default=42
        Random state for reproducibility
    svr_C : float, default=100.0
        C parameter for SVR model
    ridge_alpha : float, default=1.0
        Alpha (regularization strength) for Ridge model
    rf_n_estimators : int, default=10
        Number of estimators for Random Forest (should match filtering parameters)
    rf_max_depth : int, default=3
        Max depth for Random Forest (should match filtering parameters)
    
    Returns:
    --------
    mean_r2 : float
        Mean R² across all three models
    scores : dict
        Dictionary with individual model R² scores: {'ridge': float, 'svr': float, 'rf': float}
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # Extract features for each model
    if isinstance(X_subset, pd.DataFrame):
        X_ridge = X_subset[ridge_features].values
        X_svr = X_subset[svr_features].values
        X_rf = X_subset[rf_features].values
    else:
        if X_original is not None and isinstance(X_original, pd.DataFrame):
            ridge_idx = [X_original.columns.get_loc(f) for f in ridge_features]
            svr_idx = [X_original.columns.get_loc(f) for f in svr_features]
            rf_idx = [X_original.columns.get_loc(f) for f in rf_features]
        else:
            ridge_idx = list(range(len(ridge_features)))
            svr_idx = list(range(len(svr_features)))
            rf_idx = list(range(len(rf_features)))
        X_ridge = X_subset[:, ridge_idx]
        X_svr = X_subset[:, svr_idx]
        X_rf = X_subset[:, rf_idx]
    
    y_subset_values = y_subset.values if isinstance(y_subset, pd.Series) else y_subset
    
    if len(y_subset_values) < 10:  # Increased minimum from 5 to 10 for stability
        return np.nan, {'ridge': np.nan, 'svr': np.nan, 'rf': np.nan}
    
    # Use fewer folds for small datasets - ensure at least 3 samples per fold
    n_splits = min(5, max(3, len(y_subset_values) // 3))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Ridge
    ridge_model = Ridge(alpha=ridge_alpha)
    ridge_scores = []
    for train_idx, val_idx in cv.split(X_ridge):
        if len(val_idx) < 2:
            continue
        ridge_model.fit(X_ridge[train_idx], y_subset_values[train_idx])
        y_pred = ridge_model.predict(X_ridge[val_idx])
        ridge_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    ridge_r2 = np.mean(ridge_scores) if ridge_scores else np.nan
    
    # SVR
    svr_model = SklearnPipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=svr_C, gamma='scale', epsilon=0.01))
    ])
    svr_scores = []
    for train_idx, val_idx in cv.split(X_svr):
        if len(val_idx) < 2:
            continue
        svr_model.fit(X_svr[train_idx], y_subset_values[train_idx])
        y_pred = svr_model.predict(X_svr[val_idx])
        svr_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    svr_r2 = np.mean(svr_scores) if svr_scores else np.nan
    
    # Random Forest - use same parameters as during filtering for consistency
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    rf_scores = []
    for train_idx, val_idx in cv.split(X_rf):
        if len(val_idx) < 2:
            continue
        rf_model.fit(X_rf[train_idx], y_subset_values[train_idx])
        y_pred = rf_model.predict(X_rf[val_idx])
        rf_scores.append(r2_score(y_subset_values[val_idx], y_pred))
    rf_r2 = np.mean(rf_scores) if rf_scores else np.nan
    
    # Calculate mean only if at least one model has valid scores
    if not np.isnan(ridge_r2) or not np.isnan(svr_r2) or not np.isnan(rf_r2):
        valid_scores = [s for s in [ridge_r2, svr_r2, rf_r2] if not np.isnan(s)]
        mean_r2 = np.mean(valid_scores) if valid_scores else np.nan
    else:
        mean_r2 = np.nan
    
    return mean_r2, {'ridge': ridge_r2, 'svr': svr_r2, 'rf': rf_r2}


def get_ensemble_predictions(X_subset, y_subset, 
                             ridge_features, svr_features, rf_features,
                             X_original=None, random_state=42, svr_C=100.0,
                             ridge_alpha=1.0, rf_n_estimators=10, rf_max_depth=3):
    """
    Get cross-validation predictions from ensemble where each model uses its own selected features.
    Returns predictions for each model and the ensemble mean.
    
    Parameters:
    -----------
    X_subset : pandas.DataFrame or numpy.ndarray
        Feature subset (samples x features)
    y_subset : pandas.Series or numpy.ndarray
        Target subset
    ridge_features : list
        List of feature names/indices for Ridge model
    svr_features : list
        List of feature names/indices for SVR model
    rf_features : list
        List of feature names/indices for Random Forest model
    X_original : pandas.DataFrame or numpy.ndarray, optional
        Original X dataset (needed if X_subset is numpy array and features are named)
        If None and X_subset is DataFrame, uses X_subset
    random_state : int, default=42
        Random state for reproducibility
    svr_C : float, default=100.0
        C parameter for SVR model
    ridge_alpha : float, default=1.0
        Alpha (regularization strength) for Ridge model
    rf_n_estimators : int, default=10
        Number of estimators for Random Forest (should match filtering parameters)
    rf_max_depth : int, default=3
        Max depth for Random Forest (should match filtering parameters)
    
    Returns:
    --------
    y_pred_ridge : numpy.ndarray or pandas.Series
        Cross-validation predictions from Ridge model
    y_pred_svr : numpy.ndarray or pandas.Series
        Cross-validation predictions from SVR model
    y_pred_rf : numpy.ndarray or pandas.Series
        Cross-validation predictions from Random Forest model
    y_pred_ensemble : numpy.ndarray or pandas.Series
        Ensemble mean predictions (average of all three models)
    """
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # Extract features for each model
    if isinstance(X_subset, pd.DataFrame):
        X_ridge = X_subset[ridge_features].values
        X_svr = X_subset[svr_features].values
        X_rf = X_subset[rf_features].values
    else:
        if X_original is not None and isinstance(X_original, pd.DataFrame):
            ridge_idx = [X_original.columns.get_loc(f) for f in ridge_features]
            svr_idx = [X_original.columns.get_loc(f) for f in svr_features]
            rf_idx = [X_original.columns.get_loc(f) for f in rf_features]
        else:
            ridge_idx = list(range(len(ridge_features)))
            svr_idx = list(range(len(svr_features)))
            rf_idx = list(range(len(rf_features)))
        X_ridge = X_subset[:, ridge_idx]
        X_svr = X_subset[:, svr_idx]
        X_rf = X_subset[:, rf_idx]
    
    y_subset_values = y_subset.values if isinstance(y_subset, pd.Series) else y_subset
    
    if len(y_subset_values) < 10:  # Increased minimum from 5 to 10 for stability
        # Return None for all predictions if insufficient samples
        return None, None, None, None
    
    # Use fewer folds for small datasets - ensure at least 3 samples per fold
    n_splits = min(5, max(3, len(y_subset_values) // 3))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Ridge predictions
    ridge_model = Ridge(alpha=ridge_alpha)
    y_pred_ridge = cross_val_predict(ridge_model, X_ridge, y_subset_values, cv=cv)
    
    # SVR predictions
    svr_model = SklearnPipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=svr_C, gamma='scale', epsilon=0.01))
    ])
    y_pred_svr = cross_val_predict(svr_model, X_svr, y_subset_values, cv=cv)
    
    # Random Forest predictions - use same parameters as during filtering for consistency
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    y_pred_rf = cross_val_predict(rf_model, X_rf, y_subset_values, cv=cv)
    
    # Ensemble mean predictions
    y_pred_ensemble = np.mean([y_pred_ridge, y_pred_svr, y_pred_rf], axis=0)
    
    # Preserve index if y_subset is a Series
    if isinstance(y_subset, pd.Series):
        y_pred_ridge = pd.Series(y_pred_ridge, index=y_subset.index)
        y_pred_svr = pd.Series(y_pred_svr, index=y_subset.index)
        y_pred_rf = pd.Series(y_pred_rf, index=y_subset.index)
        y_pred_ensemble = pd.Series(y_pred_ensemble, index=y_subset.index)
    
    return y_pred_ridge, y_pred_svr, y_pred_rf, y_pred_ensemble