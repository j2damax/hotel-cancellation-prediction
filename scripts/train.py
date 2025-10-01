"""
Training script for hotel cancellation prediction models.
Trains LogReg, Random Forest, XGBoost, and PyTorch MLP using MLflow.
"""

import os
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


def generate_sample_data(n_samples=10000):
    """Generate sample hotel booking data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    lead_time = np.random.randint(0, 365, n_samples)
    arrival_month = np.random.randint(1, 13, n_samples)
    stays_weekend = np.random.randint(0, 10, n_samples)
    stays_week = np.random.randint(0, 20, n_samples)
    adults = np.random.randint(1, 5, n_samples)
    children = np.random.randint(0, 4, n_samples)
    is_repeated = np.random.binomial(1, 0.3, n_samples)
    previous_cancellations = np.random.poisson(0.2, n_samples)
    booking_changes = np.random.poisson(0.3, n_samples)
    adr = np.random.uniform(50, 300, n_samples)
    required_parking = np.random.binomial(1, 0.2, n_samples)
    special_requests = np.random.randint(0, 6, n_samples)
    
    # Generate target with some logic
    cancellation_prob = (
        0.1 * (lead_time > 100) +
        0.15 * (previous_cancellations > 0) +
        0.1 * (booking_changes > 1) +
        0.05 * (required_parking == 0) +
        0.1 * (is_repeated == 0) +
        0.1
    )
    is_canceled = np.random.binomial(1, np.clip(cancellation_prob, 0, 1), n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'stays_weekend_nights': stays_weekend,
        'stays_week_nights': stays_week,
        'adults': adults,
        'children': children,
        'is_repeated_guest': is_repeated,
        'previous_cancellations': previous_cancellations,
        'booking_changes': booking_changes,
        'adr': adr,
        'required_car_parking_spaces': required_parking,
        'total_of_special_requests': special_requests,
        'is_canceled': is_canceled
    })
    
    return data


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


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model with MLflow tracking."""
    with mlflow.start_run(run_name="LogisticRegression"):
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        
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
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"LogisticRegression - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
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
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"RandomForest - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
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
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        print(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
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
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
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
        mlflow.pytorch.log_model(model, "model")
        
        print(f"PyTorch_MLP - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return model


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("Hotel Cancellation Prediction - Model Training")
    print("=" * 80)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hotel_cancellation_prediction")
    
    # Generate or load data
    print("\n1. Loading data...")
    data = generate_sample_data(n_samples=10000)
    print(f"   Data shape: {data.shape}")
    print(f"   Cancellation rate: {data['is_canceled'].mean():.2%}")
    
    # Split features and target
    X = data.drop('is_canceled', axis=1)
    y = data['is_canceled']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("\n2. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save scaler
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("   Scaler saved to models/scaler.pkl")
    
    # Train models
    print("\n3. Training models...")
    print("-" * 80)
    
    print("\n   Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n   Training Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n   Training XGBoost...")
    xgb_model = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n   Training PyTorch MLP...")
    mlp_model = train_pytorch_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n" + "=" * 80)
    print("Training completed! Check MLflow UI with: mlflow ui")
    print("=" * 80)


if __name__ == "__main__":
    main()
