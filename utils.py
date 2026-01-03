import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
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
        'predictions': y_pred,
        'model': model,
        'model_name': model_name,
        'cv': cv_generator
    }
    
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
        
        # Fit model on initial representative samples
        X_initial = X_selected[initial_indices, :]
        y_initial = y_values[initial_indices]
        
        if len(np.unique(y_initial)) > 1:  # Need variation in y
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_initial, y_initial)
            y_pred_initial = model.predict(X_initial)
            best_r2 = r2_score(y_initial, y_pred_initial)
            # Calculate correlation
            if len(y_initial) > 1 and np.std(y_initial) > 0 and np.std(y_pred_initial) > 0:
                best_correlation = np.corrcoef(y_initial, y_pred_initial)[0, 1]
            else:
                best_correlation = 0.0
        else:
            # Fallback: use first n_initial samples
            initial_indices = list(range(min(n_initial, n_samples)))
            X_initial = X_selected[initial_indices, :]
            y_initial = y_values[initial_indices]
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_initial, y_initial)
            y_pred_initial = model.predict(X_initial)
            best_r2 = r2_score(y_initial, y_pred_initial)
            best_correlation = 0.0
        
        selected_sample_indices = initial_indices.copy()
        remaining_indices = [i for i in remaining_indices if i not in selected_sample_indices]
        
        print(f"Initial {len(selected_sample_indices)} samples - R2: {best_r2:.4f}, Correlation: {best_correlation:.4f}")
        print(f"Starting with samples: {selected_sample_indices}")
        
        # Step 2: Iteratively add samples that improve R2
        current_r2 = best_r2
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
                # Force add samples until minimum is met (even if R2 doesn't improve much)
                # Find the best remaining sample
                best_candidate = None
                best_candidate_r2 = current_r2
                
                for candidate_idx in remaining_indices:
                    test_indices = selected_sample_indices + [candidate_idx]
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    
                    model = Ridge(alpha=1.0, random_state=random_state)
                    model.fit(X_test, y_test)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_candidate_r2:
                        best_candidate_r2 = r2
                        best_candidate = candidate_idx
                
                # Always add a sample if we're below minimum, even if R2 doesn't improve
                if best_candidate is None and len(remaining_indices) > 0:
                    # Take the first remaining sample if no improvement found
                    best_candidate = remaining_indices[0]
                    test_indices = selected_sample_indices + [best_candidate]
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    model = Ridge(alpha=1.0, random_state=random_state)
                    model.fit(X_test, y_test)
                    y_pred = model.predict(X_test)
                    best_candidate_r2 = r2_score(y_test, y_pred)
                
                if best_candidate is not None:
                    selected_sample_indices.append(best_candidate)
                    remaining_indices.remove(best_candidate)
                    # Recalculate R2 and correlation for updated set
                    test_indices = selected_sample_indices
                    X_test = X_selected[test_indices, :]
                    y_test = y_values[test_indices]
                    model = Ridge(alpha=1.0, random_state=random_state)
                    model.fit(X_test, y_test)
                    y_pred = model.predict(X_test)
                    best_candidate_r2 = r2_score(y_test, y_pred)
                    if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                        best_candidate_correlation = np.corrcoef(y_test, y_pred)[0, 1]
                    else:
                        best_candidate_correlation = 0.0
                    improvement = best_candidate_r2 - current_r2
                    current_r2 = best_candidate_r2
                    current_correlation = best_candidate_correlation
                    improvement_count = 0
                    print(f"Added sample {best_candidate} (to meet minimum): R2 = {current_r2:.4f}, Correlation = {current_correlation:.4f} (improvement: {improvement:.4f})")
                    continue
            best_candidate = None
            best_candidate_r2 = current_r2
            best_candidate_idx = None
            
            # Try each remaining sample
            for candidate_idx in remaining_indices:
                # Add this candidate to selected samples
                test_indices = selected_sample_indices + [candidate_idx]
                X_test = X_selected[test_indices, :]
                y_test = y_values[test_indices]
                
                # Fit model
                model = Ridge(alpha=1.0, random_state=random_state)
                model.fit(X_test, y_test)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                # Check if this improves R2
                if r2 > best_candidate_r2 + min_r2_improvement:
                    best_candidate_r2 = r2
                    best_candidate = candidate_idx
                    best_candidate_idx = candidate_idx
            
            # Add the best candidate if it improves R2
            if best_candidate is not None:
                selected_sample_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
                # Recalculate R2 and correlation for updated set
                test_indices = selected_sample_indices
                X_test = X_selected[test_indices, :]
                y_test = y_values[test_indices]
                model = Ridge(alpha=1.0, random_state=random_state)
                model.fit(X_test, y_test)
                y_pred = model.predict(X_test)
                best_candidate_r2 = r2_score(y_test, y_pred)
                if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                    best_candidate_correlation = np.corrcoef(y_test, y_pred)[0, 1]
                else:
                    best_candidate_correlation = 0.0
                improvement = best_candidate_r2 - current_r2
                current_r2 = best_candidate_r2
                current_correlation = best_candidate_correlation
                improvement_count = 0
                print(f"Added sample {best_candidate}: R2 = {current_r2:.4f}, Correlation = {current_correlation:.4f} (improvement: {improvement:.4f})")
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
                        
                        model = Ridge(alpha=1.0, random_state=random_state)
                        model.fit(X_test, y_test)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                            correlation = np.corrcoef(y_test, y_pred)[0, 1]
                        else:
                            correlation = 0.0
                        
                        selected_sample_indices.append(candidate_idx)
                        remaining_indices.remove(candidate_idx)
                        current_r2 = r2
                        current_correlation = correlation
                        improvement_count = 0
                        print(f"Added sample {candidate_idx} (forced to meet minimum): R2 = {current_r2:.4f}, Correlation = {current_correlation:.4f}")
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
            
            model = Ridge(alpha=1.0, random_state=random_state)
            model.fit(X_test, y_test)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if len(y_test) > 1 and np.std(y_test) > 0 and np.std(y_pred) > 0:
                correlation = np.corrcoef(y_test, y_pred)[0, 1]
            else:
                correlation = 0.0
            
            selected_sample_indices.append(candidate_idx)
            remaining_indices.remove(candidate_idx)
            current_r2 = r2
            current_correlation = correlation
            print(f"Added sample {candidate_idx} (post-loop, to meet minimum): R2 = {current_r2:.4f}, Correlation = {current_correlation:.4f}")
        
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
        
        print(f"\nFinal selection: {len(selected_sample_indices)} samples")
        print(f"  Training R2 = {filtered_r2_train:.4f} (may be overfitted)")
        print(f"  Training Correlation = {filtered_correlation:.4f}")
        if cv_scores_filtered is not None:
            print(f"  Cross-validated R2 = {filtered_r2_cv:.4f} ± {np.std(cv_scores_filtered):.4f}")
        else:
            print(f"  Cross-validated R2 = {filtered_r2_cv:.4f} (too few samples for CV)")
        
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
