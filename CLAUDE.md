# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning pipeline for predicting glioma patient survival (`Sobrevida_dias`) from TCGA RNA-seq gene expression data. It implements several sample-filtering strategies to improve model generalization on a noisy clinical dataset.

## Running Scripts

```bash
# Cluster-guided Ridge regression pipeline (primary pipeline)
python main_cluster_ridge.py

# Multi-model ensemble filtering (Ridge + SVR + Random Forest)
python main.py

# SVR-only filtering and evaluation
python main_svr.py

# Interactive exploration
jupyter notebook notebook.ipynb
jupyter notebook ridge_analysis_explained.ipynb
```

All scripts must be run from the repository root (paths like `data/` and `output/` are relative).

## Environment

Python 3.11, scikit-learn 1.5.1 (HDBSCAN is built-in, no separate package needed), pandas 2.2, numpy 1.26, scipy 1.13, matplotlib 3.9, seaborn 0.13. No `requirements.txt` exists — install manually.

## Architecture

### Data Layer

- `data/ClinicaGliomasDic2025verificados.csv` — clinical metadata; key columns: `TCGACode` (patient ID), `Sobrevida_dias` (survival days, regression target)
- `data/TCGAGliomas_RNAm_Filtrado_QC_verif.csv` — full QC-filtered gene expression (genes as rows, TCGACodes as columns)
- `data/TCGAGliomas_RNAm_Filtrado_QC_DEGCol_verif.csv` — DEG-filtered subset
- `data/TCGAGliomas_RNAm_Filtrado_QC_correlacionCol_verif.csv` — correlation-filtered subset

`load_data()` in `utils.py` handles alignment: it transposes the gene matrix so samples are rows, then inner-joins on `TCGACode` with clinical data.

### Core Library (`utils.py`)

All reusable logic lives here. Key functions:

| Function | Purpose |
|---|---|
| `load_data()` | Load + align clinical and gene expression data |
| `cross_validate_regression()` | KFold CV returning R², correlation, and OOF predictions |
| `filter_samples_for_model_with_features()` | Greedy sample removal: iteratively drops the sample whose removal most improves CV R², up to `max_outliers_to_remove` |
| `filter_data_for_linear_model()` | Feature selection + linear filtering variant |
| `augment_regression_data()` | Gaussian noise / mixup / SMOTE-like augmentation |
| `cross_validate_regression_with_augmentation()` | CV with per-fold augmentation |
| `evaluate_ensemble()` / `evaluate_ensemble_with_model_features()` | Ensemble of Ridge + SVR + RF on a filtered subset |
| `binarize_y()` | Convert survival days to binary (threshold at 365 days) |
| `cross_validate_classification()` / `cross_validate_classification_with_feature_selection()` | Classification CV with optional feature selection inside each fold |

### Pipeline Scripts

**`main_cluster_ridge.py`** — Most sophisticated pipeline:
1. Log-transform survival (`log(1 + Sobrevida_dias)`)
2. F-regression feature selection (top 1000 genes)
3. PCA (30 components) → HDBSCAN clustering
4. Per-cluster Ridge CV evaluation
5. Greedy cluster removal maximizing `R²_cv − λ × fraction_removed` (λ=0.5 default)
6. Ridge regression on retained samples
7. Logistic Regression + Random Forest classifiers to characterize retained vs. discarded samples
8. Saves all results to `output/`

Configuration constants are at the top of the file (e.g., `N_FEATURES`, `RIDGE_ALPHA`, `LAMBDA_PENALTY`, `HDBSCAN_MIN_CLUSTER`).

**`main.py`** — Runs `filter_samples_for_model_with_features()` independently for Ridge, SVR, and Random Forest, then evaluates an ensemble on the union/intersection of retained samples.

**`main_svr.py`** — SVR-only version of the filtering pipeline with a visualization of predicted vs. actual values.

### Outputs

All scripts write to `output/`. Typical files:
- `output/selected_features.csv` — genes selected by F-regression
- `output/sample_cluster_membership.csv` — HDBSCAN cluster assignment per sample
- `output/cluster_statistics.csv` — per-cluster R² and removal decisions
- `output/cluster_removal_log.csv` — trace of which clusters were removed and why
- `output/classifier_*.csv` — classifier performance metrics and feature importance
- `output/*.png` — scatter plots, trace curves, feature importance bar charts
- `output/old_results/` — previous run snapshots

## Key Design Decisions

- **Sample filtering philosophy**: Instead of treating all samples equally, the pipelines identify and remove samples that hurt generalization. The greedy approach in `filter_samples_for_model_with_features()` evaluates each candidate removal by running full CV, so it is O(n²) and slow for large datasets.
- **Log-transforming survival**: `main_cluster_ridge.py` applies `log(1 + y)` before modeling; `main.py` and `main_svr.py` do not — be consistent when comparing results across scripts.
- **Feature selection scope**: Feature selection in `main_cluster_ridge.py` uses F-regression on the *full* dataset before splitting; inside `cross_validate_classification_with_feature_selection()` it is done *within each fold* to avoid leakage.
