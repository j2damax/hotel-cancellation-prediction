"""
Training script for hotel cancellation prediction models.
Trains LogReg, Random Forest, XGBoost, and PyTorch MLP using MLflow.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import warnings
# Ensure project root is on path for 'src' package imports when executing as script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import PreprocessingPipeline
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    """PyTorch MLP for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def _log_model_with_compat(module, model, signature, input_example):
    """Log a model using new MLflow API (name=) with fallback to legacy artifact_path param.

    This suppresses the deprecation warning: `artifact_path` is deprecated. Please use `name` instead.
    """
    try:  # Preferred new-style
        module.log_model(model, signature=signature, input_example=input_example, name="model")
    except TypeError:  # Older MLflow fallback
        module.log_model(model, "model", signature=signature, input_example=input_example)


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model with MLflow tracking."""
    with mlflow.start_run(run_name="LogisticRegression"):
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        # Global preprocessing context if available
        if 'PREPROCESSING_CONTEXT' in globals():
            ctx = globals()['PREPROCESSING_CONTEXT']
            mlflow.log_param('categorical_strategy', ctx.get('categorical_strategy'))
            if ctx.get('categorical_strategy') == 'target':
                mlflow.log_param('target_smoothing', ctx.get('target_smoothing'))
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model with signature & example
        from mlflow.models.signature import infer_signature
        X_train_float = X_train.astype('float64')  # Ensure float schema to avoid integer missing value warnings
        signature = infer_signature(X_train_float, model.predict_proba(X_train)[:, 1])
        input_example = X_train_float.head(3)
        _log_model_with_compat(mlflow.sklearn, model, signature, input_example)
        
        print(
            "LogisticRegression - "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )
        
        return model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model with MLflow tracking."""
    with mlflow.start_run(run_name="RandomForest"):
        # Parameters
        n_estimators = 100
        max_depth = 10
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        if 'PREPROCESSING_CONTEXT' in globals():
            ctx = globals()['PREPROCESSING_CONTEXT']
            mlflow.log_param('categorical_strategy', ctx.get('categorical_strategy'))
            if ctx.get('categorical_strategy') == 'target':
                mlflow.log_param('target_smoothing', ctx.get('target_smoothing'))
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model with signature & example
        from mlflow.models.signature import infer_signature
        X_train_float = X_train.astype('float64')
        signature = infer_signature(X_train_float, model.predict_proba(X_train)[:, 1])
        input_example = X_train_float.head(3)
        _log_model_with_compat(mlflow.sklearn, model, signature, input_example)
        
        print(
            "RandomForest - "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )
        
        return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with MLflow tracking."""
    with mlflow.start_run(run_name="XGBoost"):
        # Parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        if 'PREPROCESSING_CONTEXT' in globals():
            ctx = globals()['PREPROCESSING_CONTEXT']
            mlflow.log_param('categorical_strategy', ctx.get('categorical_strategy'))
            if ctx.get('categorical_strategy') == 'target':
                mlflow.log_param('target_smoothing', ctx.get('target_smoothing'))
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model with signature & example
        from mlflow.models.signature import infer_signature
        X_train_float = X_train.astype('float64')
        signature = infer_signature(X_train_float, model.predict_proba(X_train)[:, 1])
        input_example = X_train_float.head(3)
        _log_model_with_compat(mlflow.xgboost, model, signature, input_example)
        
        print(
            "XGBoost - "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )
        
        return model


def train_pytorch_mlp(X_train, y_train, X_test, y_test):
    """Train PyTorch MLP model with MLflow tracking."""
    with mlflow.start_run(run_name="PyTorch_MLP"):
        # Parameters
        hidden_dims = [64, 32]
        dropout = 0.3
        learning_rate = 0.001
        batch_size = 64
        epochs = 50
        
        # Log parameters
        mlflow.log_param("model_type", "PyTorch_MLP")
        mlflow.log_param("hidden_dims", str(hidden_dims))
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        if 'PREPROCESSING_CONTEXT' in globals():
            ctx = globals()['PREPROCESSING_CONTEXT']
            mlflow.log_param('categorical_strategy', ctx.get('categorical_strategy'))
            if ctx.get('categorical_strategy') == 'target':
                mlflow.log_param('target_smoothing', ctx.get('target_smoothing'))
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = MLPClassifier(input_dim, hidden_dims, dropout)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        from mlflow.models.signature import infer_signature
        # Use probability outputs for signature inference
        with torch.no_grad():
            train_probs = model(torch.FloatTensor(X_train.values)).numpy().flatten()
        X_train_float = X_train.astype('float64')
        signature = infer_signature(X_train_float, train_probs)
        input_example = X_train_float.head(3)
        _log_model_with_compat(mlflow.pytorch, model, signature, input_example)
        
        print(
            "PyTorch_MLP - "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
        )
        
        return model


def load_engineered_dataset(features_path: str = 'data/processed/hotel_booking_features.csv',
                            contract_path: str = 'artifacts/feature_contract.json',
                            target: str = 'is_canceled'):
    """Load engineered dataset using feature contract for column ordering.

    Returns X (DataFrame), y (Series)
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Engineered features CSV not found: {features_path}. Run feature_engineering.py first.")
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Feature contract not found: {contract_path}")
    df = pd.read_csv(features_path)
    with open(contract_path) as f:
        contract = json.load(f)
    feature_order = contract['feature_order']
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing contract columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in engineered dataset")
    # Preserve order from contract
    X = df[feature_order].copy()
    y = df[target].copy()
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train models on synthetic or engineered dataset")
    # Removed synthetic option; engineered dataset is now required for consistency.
    parser.add_argument('--features-path', default='data/processed/hotel_booking_features.csv', help='Path to engineered features CSV')
    parser.add_argument('--contract-path', default='artifacts/feature_contract.json', help='Path to feature_contract.json')
    parser.add_argument('--target', default='is_canceled', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--no-scale', action='store_true', help='Skip scaling (e.g., for tree-only experiments)')
    parser.add_argument('--limit-rows', type=int, default=None, help='Optional row limit for faster experimentation')
    parser.add_argument('--categorical-strategy', choices=['drop', 'onehot', 'target'], default='drop', help='Categorical handling strategy: drop | onehot | target (mean target encoding).')
    parser.add_argument('--preprocessor-path', default='models/preprocessor.pkl', help='Path to save fitted preprocessing pipeline.')
    # Cross-validation controls
    parser.add_argument('--cv-folds', type=int, default=1, help='If >1, run stratified K-fold CV before holdout training.')
    parser.add_argument('--cv-include-mlp', action='store_true', help='Include PyTorch MLP in cross-validation (slower).')
    parser.add_argument('--cv-random-state', type=int, default=42, help='Random state for StratifiedKFold shuffling.')
    return parser.parse_args()


def perform_cross_validation(X: pd.DataFrame, y: pd.Series, args) -> dict:
    """Run stratified K-fold cross-validation for LR, RF, XGB (optionally MLP) with per-fold preprocessing.

    Returns a nested dict:
    {
       model_name: {
           'folds': [ {metrics...}, ... ],
           'aggregate': { 'f1_score_mean': ..., 'f1_score_std': ..., ... }
       }, ...
    }
    """
    from sklearn.model_selection import StratifiedKFold
    k = args.cv_folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.cv_random_state)
    models_to_run = ['LogisticRegression', 'RandomForest', 'XGBoost']
    if args.cv_include_mlp:
        models_to_run.append('PyTorch_MLP')

    cv_results: dict = {m: {'folds': []} for m in models_to_run}

    fold_index = 0
    for train_idx, val_idx in skf.split(X, y):
        fold_index += 1
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fresh preprocessing per fold to avoid leakage across folds
        preproc = PreprocessingPipeline(
            categorical_strategy=args.categorical_strategy,
            scale=not args.no_scale
        )
        if args.categorical_strategy == 'target':
            X_tr_proc = preproc.fit_transform(X_tr, y_tr)
        else:
            X_tr_proc = preproc.fit_transform(X_tr)
        X_val_proc = preproc.transform(X_val)

        # Train each model WITHOUT logging to MLflow inside CV
        # (We aggregate and can optionally log aggregate metrics once.)
        # Logistic Regression
        if 'LogisticRegression' in models_to_run:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_tr_proc, y_tr)
            y_pred = lr.predict(X_val_proc)
            y_proba = lr.predict_proba(X_val_proc)[:, 1]
            cv_results['LogisticRegression']['folds'].append(evaluate_model(y_val, y_pred, y_proba))
        # Random Forest
        if 'RandomForest' in models_to_run:
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_tr_proc, y_tr)
            y_pred = rf.predict(X_val_proc)
            y_proba = rf.predict_proba(X_val_proc)[:, 1]
            cv_results['RandomForest']['folds'].append(evaluate_model(y_val, y_pred, y_proba))
        # XGBoost
        if 'XGBoost' in models_to_run:
            xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': 42
            }
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_tr_proc, y_tr)
            y_pred = xgb_model.predict(X_val_proc)
            y_proba = xgb_model.predict_proba(X_val_proc)[:, 1]
            cv_results['XGBoost']['folds'].append(evaluate_model(y_val, y_pred, y_proba))
        # PyTorch MLP (optional)
        if 'PyTorch_MLP' in models_to_run:
            # Lightweight config (fewer epochs for CV speed)
            hidden_dims = [64, 32]
            dropout = 0.3
            learning_rate = 0.001
            batch_size = 64
            epochs = 20
            X_tr_tensor = torch.FloatTensor(X_tr_proc.values)
            y_tr_tensor = torch.FloatTensor(y_tr.values).reshape(-1, 1)
            X_val_tensor = torch.FloatTensor(X_val_proc.values)
            model = MLPClassifier(X_tr_proc.shape[1], hidden_dims, dropout)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            ds = TensorDataset(X_tr_tensor, y_tr_tensor)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                proba = model(X_val_tensor).numpy().flatten()
                pred = (proba > 0.5).astype(int)
            cv_results['PyTorch_MLP']['folds'].append(evaluate_model(y_val, pred, proba))

    # Aggregate statistics
    for model_name, result in cv_results.items():
        folds = result['folds']
        if not folds:
            continue
        aggregate = {}
        metric_keys = folds[0].keys()
        for mk in metric_keys:
            values = [f[mk] for f in folds]
            aggregate[f"{mk}_mean"] = float(np.mean(values))
            aggregate[f"{mk}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result['aggregate'] = aggregate
    return cv_results


def select_champion_from_cv(cv_results: dict) -> dict:
    """Select champion model based on highest f1_score_mean with roc_auc_mean tie-break.

    Returns champion record:
      { 'model_name': ..., 'f1_score_mean': ..., 'roc_auc_mean': ..., 'aggregate': {...} }
    Raises ValueError if insufficient data.
    """
    best = None
    for model_name, data in cv_results.items():
        agg = data.get('aggregate')
        if not agg or 'f1_score_mean' not in agg:
            continue
        candidate = {
            'model_name': model_name,
            'f1_score_mean': agg.get('f1_score_mean'),
            'roc_auc_mean': agg.get('roc_auc_mean'),
            'aggregate': agg
        }
        if best is None:
            best = candidate
        else:
            # Primary: F1 mean; Secondary: ROC-AUC mean
            if candidate['f1_score_mean'] > best['f1_score_mean'] + 1e-6 or (
                abs(candidate['f1_score_mean'] - best['f1_score_mean']) <= 1e-6 and candidate['roc_auc_mean'] > best['roc_auc_mean']
            ):
                best = candidate
    if best is None:
        raise ValueError("No valid aggregates to select champion.")
    return best


def generate_champion_diagnostics(model_name: str, model_obj, X_test: pd.DataFrame, y_test: pd.Series, probabilities: np.ndarray, artifacts_dir: str = 'artifacts'):
    """Generate diagnostic artifacts for the champion model.

    Artifacts:
      - confusion_matrix.png (matplotlib plot)
      - roc_curve.json (fpr, tpr, thresholds)
      - pr_curve.json (precision, recall, thresholds)
      - classification_report.json
      - threshold_sweep.csv (threshold, precision, recall, f1)
    Returns selected decision threshold (argmax F1) and its metrics dict.
    """
    import matplotlib.pyplot as plt
    os.makedirs(artifacts_dir, exist_ok=True)

    # Confusion matrix at default 0.5
    y_pred_default = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_default)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion Matrix ({model_name})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='black')
    fig.tight_layout()
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
    # Some sklearn versions include an initial inf threshold; sanitize for strict JSON and downstream tools.
    roc_thresh_list = []
    for val in roc_thresholds.tolist():
        if isinstance(val, (float, int)) and (np.isinf(val) or np.isnan(val)):
            roc_thresh_list.append(None)
        else:
            roc_thresh_list.append(val)
    roc_payload = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': roc_thresh_list
    }
    with open(os.path.join(artifacts_dir, 'roc_curve.json'), 'w') as f:
        json.dump(roc_payload, f, indent=2)

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
    pr_payload = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': pr_thresholds.tolist()
    }
    with open(os.path.join(artifacts_dir, 'pr_curve.json'), 'w') as f:
        json.dump(pr_payload, f, indent=2)

    # Classification report (default threshold)
    report = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
    with open(os.path.join(artifacts_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Threshold sweep
    sweep_rows = []
    thresholds = np.linspace(0, 1, 101)
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = {}
    for thr in thresholds:
        y_pred_thr = (probabilities >= thr).astype(int)
        prec = precision_score(y_test, y_pred_thr, zero_division=0)
        rec = recall_score(y_test, y_pred_thr, zero_division=0)
        f1 = f1_score(y_test, y_pred_thr, zero_division=0)
        sweep_rows.append({'threshold': float(thr), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1)})
        if f1 > best_f1 + 1e-12:  # strict improvement
            best_f1 = f1
            best_threshold = float(thr)
            best_metrics = {'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1)}
    import csv
    sweep_path = os.path.join(artifacts_dir, 'threshold_sweep.csv')
    with open(sweep_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['threshold','precision','recall','f1_score'])
        writer.writeheader()
        writer.writerows(sweep_rows)

    return best_threshold, best_metrics


def generate_shap_artifacts(model_name: str, model_obj, X_train_proc: pd.DataFrame, X_test_proc: pd.DataFrame, y_test: pd.Series, artifacts_dir: str = 'artifacts', sample_size: int = 200):
    """Compute SHAP values for champion model and persist artifacts.

    Artifacts:
      - shap_summary.png (beeswarm plot)
      - feature_importance.json (mean |SHAP| per feature sorted desc)
      - shap_values_sample.json (local explanations for representative samples: TP, FP, FN if identifiable)
    """
    import shap
    import matplotlib.pyplot as plt
    os.makedirs(artifacts_dir, exist_ok=True)

    # Downsample background for performance
    background = X_train_proc
    if len(background) > sample_size:
        background = background.sample(sample_size, random_state=42)

    explainer = None
    is_tree = model_name in ['XGBoost', 'RandomForest']
    is_linear = model_name == 'LogisticRegression'
    is_mlp = model_name == 'PyTorch_MLP'

    try:
        if is_tree:
            explainer = shap.TreeExplainer(model_obj)
        elif is_linear:
            # Updated: 'feature_dependence' deprecated; use 'feature_perturbation'
            explainer = shap.LinearExplainer(model_obj, background, feature_perturbation="interventional")
        elif is_mlp:
            warnings.warn("SHAP for PyTorch MLP not implemented; skipping.")
            return False
        else:
            warnings.warn(f"SHAP not supported for model {model_name}; skipping.")
            return False

        shap_values = explainer.shap_values(X_test_proc)
        if isinstance(shap_values, list):
            shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_array = shap_values

        mean_abs = np.abs(shap_array).mean(axis=0)
        importance = [
            {'feature': feat, 'mean_abs_shap': float(val)}
            for feat, val in sorted(zip(X_test_proc.columns, mean_abs), key=lambda x: x[1], reverse=True)
        ]
        with open(os.path.join(artifacts_dir, 'feature_importance.json'), 'w') as f:
            json.dump(importance, f, indent=2)

        # Beeswarm plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_array, X_test_proc, show=False, plot_type='dot')
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, 'shap_summary.png'), dpi=150)
        plt.close()

        # Bar plot
        plt.figure(figsize=(8, 6))
        top_k = 25 if len(importance) > 25 else len(importance)
        imp_slice = importance[:top_k]
        plt.barh([x['feature'] for x in reversed(imp_slice)], [x['mean_abs_shap'] for x in reversed(imp_slice)])
        plt.xlabel('Mean |SHAP| Value')
        plt.title(f'SHAP Feature Importance ({model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, 'shap_importance_bar.png'), dpi=150)
        plt.close()

        # Local explanations
        local_payload = []
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X_test_proc)[:, 1]
        else:
            probs = None
        preds = (probs >= 0.5).astype(int) if probs is not None else model_obj.predict(X_test_proc)
        y_true = y_test.values
        categories = {
            'true_positive': (preds == 1) & (y_true == 1),
            'false_positive': (preds == 1) & (y_true == 0),
            'false_negative': (preds == 0) & (y_true == 1)
        }
        for label, mask in categories.items():
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            sel = idxs[0]
            local_payload.append({
                'category': label,
                'index': int(sel),
                'y_true': int(y_true[sel]),
                'prediction': int(preds[sel]),
                'probability': float(probs[sel]) if probs is not None else None,
                'shap_values': {feat: float(val) for feat, val in zip(X_test_proc.columns, shap_array[sel])}
            })
        if not local_payload:
            for sel in range(min(3, len(X_test_proc))):
                local_payload.append({
                    'category': 'sample',
                    'index': int(sel),
                    'y_true': int(y_true[sel]),
                    'shap_values': {feat: float(val) for feat, val in zip(X_test_proc.columns, shap_array[sel])}
                })
        with open(os.path.join(artifacts_dir, 'shap_values_sample.json'), 'w') as f:
            json.dump(local_payload, f, indent=2)
        return True
    except Exception as e:
        warnings.warn(f"Failed to compute SHAP values for {model_name}: {e}")
        return False


def build_feature_name_map(preprocessor: PreprocessingPipeline, output_path: str = 'artifacts/feature_name_map.json'):
    """Construct a human-readable mapping for engineered / encoded feature names.

    Heuristics:
      - *_target_encoded -> '<base> (target encoded)'
      - *_te -> '<base> (target encoded)'
      - total_stay_duration -> 'Total stay duration (nights)'
      - total_guests -> 'Total guests (adults+children+babies)'
      - is_family -> 'Family booking flag'
      - is_peak_season / is_summer_peak / is_holiday_season -> seasonal flags
      - Otherwise snake_case -> Title Case
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping = {}
    feature_list = preprocessor.state.feature_order if preprocessor.state else []
    for feat in feature_list:
        human = feat
        if feat.endswith('_target_encoded') or feat.endswith('_te'):
            base = feat.replace('_target_encoded', '').replace('_te', '')
            human = f"{base.replace('_', ' ').title()} (target encoded)"
        elif feat == 'total_stay_duration':
            human = 'Total stay duration (nights)'
        elif feat == 'total_guests':
            human = 'Total guests (adults + children + babies)'
        elif feat == 'is_family':
            human = 'Family booking flag'
        elif feat == 'is_peak_season':
            human = 'Peak season flag'
        elif feat == 'is_summer_peak':
            human = 'Summer peak season flag'
        elif feat == 'is_holiday_season':
            human = 'Holiday season flag'
        else:
            # Generic Title Case transform
            human = feat.replace('_', ' ').title()
        mapping[feat] = human
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    return mapping


def main():
    """Main training pipeline."""
    args = parse_args()
    print("=" * 80)
    print("Hotel Cancellation Prediction - Model Training")
    print("=" * 80)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hotel_cancellation_prediction")
    
    # Generate or load data
    print("\n1. Loading data...")
    X, y = load_engineered_dataset(args.features_path, args.contract_path, args.target)
    if args.limit_rows:
        X = X.head(args.limit_rows)
        y = y.head(args.limit_rows)
    print(f"   Loaded engineered dataset: X={X.shape}, y={y.shape}")

    # Optional Cross-Validation Phase
    if args.cv_folds and args.cv_folds > 1:
        print(f"\n[CV] Running stratified {args.cv_folds}-fold cross-validation...")
        cv_results = perform_cross_validation(X, y, args)
        os.makedirs('artifacts', exist_ok=True)
        cv_path = 'artifacts/cv_metrics.json'
        with open(cv_path, 'w') as f:
            json.dump({
                'folds': args.cv_folds,
                'categorical_strategy': args.categorical_strategy,
                'include_mlp': args.cv_include_mlp,
                'results': cv_results,
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }, f, indent=2)
        print(f"[CV] Metrics written -> {cv_path}")
        # Brief summary to console (primary metric F1)
        for model_name, data in cv_results.items():
            agg = data.get('aggregate', {})
            if agg:
                print(f"   {model_name}: F1={agg.get('f1_score_mean'):.4f} ± {agg.get('f1_score_std'):.4f} | ROC-AUC={agg.get('roc_auc_mean'):.4f} ± {agg.get('roc_auc_std'):.4f}")
        print("[CV] Completed. Proceeding to hold-out training & MLflow logging...")

        # Champion selection
        try:
            champion = select_champion_from_cv(cv_results)
            print(f"[CV] Champion selected -> {champion['model_name']} (F1={champion['f1_score_mean']:.4f}, ROC-AUC={champion['roc_auc_mean']:.4f})")
            champion_meta = {
                'selection_metric': 'f1_score_mean',
                'tie_breaker': 'roc_auc_mean',
                'model_name': champion['model_name'],
                'aggregate': champion['aggregate'],
                'cv_folds': args.cv_folds,
                'timestamp': pd.Timestamp.utcnow().isoformat(),
                'notes': 'Model will be (re)trained on training split below; final persisted champion artifact occurs after training.'
            }
            with open('artifacts/champion_meta.json', 'w') as f:
                json.dump(champion_meta, f, indent=2)
            print("[CV] Champion metadata written -> artifacts/champion_meta.json")
        except ValueError as e:
            print(f"[CV] Champion selection skipped: {e}")

        # Log aggregate CV metrics to MLflow (one run for transparency)
        with mlflow.start_run(run_name="CV_Aggregates"):
            mlflow.log_param('cv_folds', args.cv_folds)
            mlflow.log_param('cv_categorical_strategy', args.categorical_strategy)
            mlflow.log_param('cv_include_mlp', args.cv_include_mlp)
            for model_name, data in cv_results.items():
                agg = data.get('aggregate', {})
                for metric_key, value in agg.items():
                    # namespaced metric key: <model>/<metric>
                    mlflow.log_metric(f"cv::{model_name}/{metric_key}", value)
            mlflow.log_artifact(cv_path)
            if os.path.exists('artifacts/champion_meta.json'):
                mlflow.log_artifact('artifacts/champion_meta.json')
            print("[CV] Aggregates logged to MLflow run 'CV_Aggregates'.")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Centralized preprocessing
    print("\n2. Preprocessing features with centralized pipeline...")
    preprocessor = PreprocessingPipeline(
        categorical_strategy=args.categorical_strategy,
        scale=not args.no_scale
    )
    # For target encoding strategy we must pass y
    if args.categorical_strategy == 'target':
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
    else:
        X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    preprocessor.save(args.preprocessor_path)
    print(f"   Preprocessor saved -> {args.preprocessor_path} | Remaining features: {X_train_processed.shape[1]}")
    # Build human-readable feature name mapping artifact
    try:
        build_feature_name_map(preprocessor)
        print("   Feature name map generated -> artifacts/feature_name_map.json")
    except Exception as e:
        print(f"   Warning: failed to build feature name map: {e}")
    # Expose minimal preprocessing context globally for model trainers to log
    globals()['PREPROCESSING_CONTEXT'] = {
        'categorical_strategy': preprocessor.categorical_strategy,
        'target_smoothing': getattr(preprocessor, 'target_smoothing', None)
    }
    if preprocessor.state and preprocessor.state.dropped_columns:
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/dropped_columns.json', 'w') as f:
            json.dump({
                'timestamp': pd.Timestamp.utcnow().isoformat(),
                'categorical_strategy': preprocessor.state.categorical_strategy,
                'dropped_columns': preprocessor.state.dropped_columns,
                'remaining_feature_count': len(preprocessor.state.feature_order)
            }, f, indent=2)
        print("   Dropped columns artifact saved -> artifacts/dropped_columns.json")
    
    # Train models
    print("\n3. Training models...")
    print("-" * 80)
    
    print("\n   Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_processed, y_train, X_test_processed, y_test)
    
    print("\n   Training Random Forest...")
    rf_model = train_random_forest(X_train_processed, y_train, X_test_processed, y_test)
    
    print("\n   Training XGBoost...")
    xgb_model = train_xgboost(X_train_processed, y_train, X_test_processed, y_test)
    
    print("\n   Training PyTorch MLP...")
    mlp_model = train_pytorch_mlp(X_train_processed, y_train, X_test_processed, y_test)
    
    print("\n" + "=" * 80)
    print("Training completed! Check MLflow UI with: mlflow ui")
    print("=" * 80)

    # Persist champion model artifact if champion_meta exists
    champion_meta_path = 'artifacts/champion_meta.json'
    if os.path.exists(champion_meta_path):
        try:
            with open(champion_meta_path) as f:
                champion_meta = json.load(f)
            model_name = champion_meta.get('model_name')
            model_obj = None
            holdout_metrics = {}
            if model_name == 'XGBoost':
                model_obj = xgb_model
                # gather metrics from its MLflow run not trivial here; recompute on test set
                y_pred = model_obj.predict(X_test_processed)
                y_proba = model_obj.predict_proba(X_test_processed)[:,1]
                holdout_metrics = evaluate_model(y_test, y_pred, y_proba)
            elif model_name == 'RandomForest':
                model_obj = rf_model
                y_pred = model_obj.predict(X_test_processed)
                y_proba = model_obj.predict_proba(X_test_processed)[:,1]
                holdout_metrics = evaluate_model(y_test, y_pred, y_proba)
            elif model_name == 'LogisticRegression':
                model_obj = lr_model
                y_pred = model_obj.predict(X_test_processed)
                y_proba = model_obj.predict_proba(X_test_processed)[:,1]
                holdout_metrics = evaluate_model(y_test, y_pred, y_proba)
            elif model_name == 'PyTorch_MLP':
                model_obj = mlp_model
                with torch.no_grad():
                    y_proba = mlp_model(torch.FloatTensor(X_test_processed.values)).numpy().flatten()
                y_pred = (y_proba > 0.5).astype(int)
                holdout_metrics = evaluate_model(y_test, y_pred, y_proba)
            else:
                print(f"[Champion] Unknown model name '{model_name}', skipping champion persistence.")
            if model_obj is not None:
                os.makedirs('models', exist_ok=True)
                champion_path = 'models/champion_model.pkl'
                try:
                    import joblib
                    joblib.dump(model_obj, champion_path)
                    champion_meta['persisted_path'] = champion_path
                    champion_meta['holdout_metrics'] = holdout_metrics
                    champion_meta['holdout_timestamp'] = pd.Timestamp.utcnow().isoformat()
                    with open(champion_meta_path, 'w') as f:
                        json.dump(champion_meta, f, indent=2)
                    print(f"[Champion] Persisted champion model -> {champion_path}")
                except Exception as e:
                    print(f"[Champion] Failed to persist champion model: {e}")
        except Exception as e:
            print(f"[Champion] Error during champion persistence: {e}")
    else:
        print("[Champion] No champion_meta.json found; skipping champion persistence.")

    # Generate diagnostics for champion (requires persisted champion and probabilities)
    if os.path.exists('models/champion_model.pkl') and os.path.exists('artifacts/champion_meta.json'):
        try:
            import joblib
            with open('artifacts/champion_meta.json') as f:
                champion_meta = json.load(f)
            champ_name = champion_meta.get('model_name')
            champion_model = joblib.load('models/champion_model.pkl')
            # Compute probabilities using champion model (relying on earlier processed test set still in scope)
            if champ_name == 'PyTorch_MLP':
                with torch.no_grad():
                    probabilities = champion_model(torch.FloatTensor(X_test_processed.values)).numpy().flatten()
            else:
                probabilities = champion_model.predict_proba(X_test_processed)[:,1]
            best_threshold, best_thr_metrics = generate_champion_diagnostics(
                champ_name, champion_model, X_test_processed, y_test, probabilities, artifacts_dir='artifacts'
            )
            champion_meta['decision_threshold'] = best_threshold
            champion_meta['decision_threshold_metrics'] = best_thr_metrics
            champion_meta['diagnostics_generated'] = pd.Timestamp.utcnow().isoformat()
            with open('artifacts/champion_meta.json', 'w') as f:
                json.dump(champion_meta, f, indent=2)
            print(f"[Diagnostics] Generated champion diagnostics. Optimal F1 threshold={best_threshold:.2f} (F1={best_thr_metrics['f1_score']:.4f})")
        except Exception as e:
            print(f"[Diagnostics] Failed to generate diagnostics: {e}")

    # SHAP interpretability for champion (only for supported model types)
    if os.path.exists('models/champion_model.pkl') and os.path.exists('artifacts/champion_meta.json'):
        try:
            import joblib
            with open('artifacts/champion_meta.json') as f:
                champion_meta = json.load(f)
            champ_name = champion_meta.get('model_name')
            champion_model = joblib.load('models/champion_model.pkl')
            shap_success = generate_shap_artifacts(
                champ_name, champion_model, X_train_processed, X_test_processed, y_test, artifacts_dir='artifacts'
            )
            if shap_success:
                champion_meta['shap_generated'] = True
                champion_meta['shap_timestamp'] = pd.Timestamp.utcnow().isoformat()
                with open('artifacts/champion_meta.json', 'w') as f:
                    json.dump(champion_meta, f, indent=2)
                print("[SHAP] Global & local SHAP artifacts generated.")
            else:
                print("[SHAP] SHAP generation skipped or failed for this model type.")
        except Exception as e:
            print(f"[SHAP] Error during SHAP generation: {e}")


if __name__ == "__main__":
    main()
