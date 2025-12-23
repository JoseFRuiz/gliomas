import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline


def load_data(clinical_path='data/ClinicaGliomasDic2025.csv', 
              gene_tpm_path='data/gene_tpm__GeneSymbol_reclasificado_TCGA_filtrado.csv'):
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


def cross_validate_regression(X, y, model=None, cv=5, scoring='r2', random_state=42):
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
        'cv': cv_generator
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


def cross_validate_classification(X, y_binary, model=None, cv=5, scoring='roc_auc', random_state=42):
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
    
    # Create cross-validation generator
    if isinstance(cv, int):
        cv_generator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
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
        'cv': cv_generator
    }
    
    return results


def cross_validate_classification_with_feature_selection(X, y_binary, model=None, cv=5, 
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
    
    # Create cross-validation generator
    if isinstance(cv, int):
        cv_generator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
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
        'cv': cv_generator
    }
    
    return results
