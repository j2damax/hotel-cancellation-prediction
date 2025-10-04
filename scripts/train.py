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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


## NOTE: generate_sample_data removed. Training now requires engineered dataset to promote reproducibility and parity with API inference.


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
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
