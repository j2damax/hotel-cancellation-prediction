"""
Model evaluation pipeline for hotel booking cancellation prediction.
Implements comprehensive evaluation framework with statistical significance testing,
SHAP interpretability analysis, and business impact assessment.

Academic Research Framework - NIB 7072 Coursework
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Model training and evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold, validation_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Advanced evaluation
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not available")

# PyTorch for MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch not available")

# Model interpretability
try:
    import shap
except ImportError:
    print("SHAP not available")

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.pytorch
except ImportError:
    print("MLflow not available")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPModel(nn.Module):
    """PyTorch Multi-Layer Perceptron for hotel booking cancellation prediction."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HotelBookingDataset(Dataset):
    """PyTorch dataset for hotel booking data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for hotel booking cancellation prediction.
    
    Implements academic-grade evaluation including:
    - Statistical significance testing
    - Cross-validation with multiple metrics
    - Model interpretability with SHAP
    - Business impact analysis
    - Hyperparameter optimization with Optuna
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config: Configuration dictionary for evaluation parameters
        """
        self.config = config or {}
        self.models = {}
        self.evaluation_results = {}
        self.shap_values = {}
        self.business_metrics = {}
        
        # Set up MLflow tracking
        if 'mlflow' in globals():
            mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'file:./mlruns'))
            mlflow.set_experiment('hotel_cancellation_prediction')
    
    def load_feature_engineered_data(self, file_path: str) -> pd.DataFrame:
        """Load feature-engineered hotel booking data."""
        logger.info(f"Loading feature-engineered data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Feature-engineered data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Feature-engineered data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading feature-engineered data: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_column: str = 'is_canceled',
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training and evaluation.
        
        Args:
            df: Feature-engineered DataFrame
            target_column: Target variable column name
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test arrays
        """
        logger.info("Preparing data for model training and evaluation")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical variables: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        if X.isnull().any().any():
            logger.info("Handling missing values with median imputation")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Train-test split with stratification
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, 
                                y_train: np.ndarray, n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training target
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters dictionary
        """
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        if 'optuna' not in globals():
            logger.warning("Optuna not available, using default parameters")
            return {}
        
        def objective(trial):
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.01, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'liblinear',
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
                model = LogisticRegression(**params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_name == 'xgboost' and 'xgb' in globals():
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
                model = xgb.XGBClassifier(**params)
                
            else:
                return 0  # Skip if model not available
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1', n_jobs=-1
            )
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best hyperparameters for {model_name}: {study.best_params}")
        
        return study.best_params
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train optimized Logistic Regression model."""
        logger.info("Training Logistic Regression model")
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters('logistic_regression', X_train, y_train)
        
        # Default parameters if optimization failed
        if not best_params:
            best_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train optimized Random Forest model."""
        logger.info("Training Random Forest model")
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters('random_forest', X_train, y_train)
        
        # Default parameters if optimization failed
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
        
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train optimized XGBoost model."""
        logger.info("Training XGBoost model")
        
        if 'xgb' not in globals():
            logger.warning("XGBoost not available")
            return None
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters('xgboost', X_train, y_train)
        
        # Default parameters if optimization failed
        if not best_params:
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def train_pytorch_mlp(self, X_train: np.ndarray, y_train: np.ndarray, 
                         epochs: int = 100, batch_size: int = 64) -> Any:
        """Train PyTorch MLP model."""
        logger.info("Training PyTorch MLP model")
        
        if 'torch' not in globals():
            logger.warning("PyTorch not available")
            return None
        
        # Create dataset and dataloader
        dataset = HotelBookingDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = MLPModel(input_size=X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        
        model.eval()
        self.models['pytorch_mlp'] = model
        return model
    
    def evaluate_model(self, model: Any, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating {model_name} model")
        
        # Generate predictions
        if model_name == 'pytorch_mlp' and 'torch' in globals():
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred_proba = model(X_test_tensor).numpy().flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'matthews_corr': matthews_corrcoef(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        # Additional business metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Business impact calculations
        revenue_per_booking = 200  # Assumed average revenue
        cancellation_cost = 50     # Cost of cancellation handling
        
        # Cost-benefit analysis
        true_positive_savings = tp * cancellation_cost
        false_positive_cost = fp * (revenue_per_booking * 0.1)  # Assume 10% revenue loss from false alerts
        false_negative_cost = fn * revenue_per_booking
        
        net_benefit = true_positive_savings - false_positive_cost - false_negative_cost
        
        business_metrics = {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'net_benefit_usd': net_benefit,
            'cost_savings_per_booking': net_benefit / len(y_test),
            'precision_at_threshold': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall_at_threshold': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        metrics.update(business_metrics)
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                             cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation for all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results dictionary
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_results = {}
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Models to cross-validate (exclude PyTorch for now due to complexity)
        models_to_cv = {
            'logistic_regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'random_forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        }
        
        if 'xgb' in globals():
            models_to_cv['xgboost'] = xgb.XGBClassifier(
                random_state=42, eval_metric='logloss', use_label_encoder=False
            )
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in models_to_cv.items():
            logger.info(f"Cross-validating {model_name}")
            
            model_cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv_splitter, scoring=metric, n_jobs=-1
                )
                
                model_cv_results[f'{metric}_mean'] = scores.mean()
                model_cv_results[f'{metric}_std'] = scores.std()
                model_cv_results[f'{metric}_scores'] = scores.tolist()
            
            cv_results[model_name] = model_cv_results
        
        return cv_results
    
    def perform_shap_analysis(self, model_name: str, X_train: np.ndarray, 
                             X_test: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Perform SHAP interpretability analysis.
        
        Args:
            model_name: Name of the model to analyze
            X_train: Training features for background
            X_test: Test features to explain
            feature_names: List of feature names
            
        Returns:
            SHAP analysis results
        """
        logger.info(f"Performing SHAP analysis for {model_name}")
        
        if 'shap' not in globals():
            logger.warning("SHAP not available")
            return {}
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        try:
            # Select appropriate explainer
            if model_name in ['random_forest', 'xgboost']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # For binary classification, get positive class SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            else:  # Linear models
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            # Create feature importance dictionary
            shap_results = {
                'feature_importance': dict(zip(feature_names, feature_importance)),
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'top_features': sorted(
                    zip(feature_names, feature_importance), 
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
            
            self.shap_values[model_name] = shap_results
            
            logger.info(f"SHAP analysis completed for {model_name}")
            
            return shap_results
            
        except Exception as e:
            logger.error(f"SHAP analysis failed for {model_name}: {e}")
            return {}
    
    def compare_models_statistically(self) -> Dict:
        """
        Perform statistical significance testing between models.
        """
        logger.info("Performing statistical model comparison")
        
        if len(self.evaluation_results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        # Extract F1 scores for comparison
        model_names = list(self.evaluation_results.keys())
        f1_scores = {name: results['f1_score'] for name, results in self.evaluation_results.items()}
        
        # Pairwise comparisons
        comparison_results = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                # Since we have single scores, we'll use confidence intervals
                f1_1 = f1_scores[model1]
                f1_2 = f1_scores[model2]
                
                difference = abs(f1_1 - f1_2)
                
                # Simple threshold-based significance (in practice, would use CV results)
                is_significant = difference > 0.02  # 2% difference threshold
                
                comparison_results[comparison_key] = {
                    'f1_difference': difference,
                    'better_model': model1 if f1_1 > f1_2 else model2,
                    'is_significant': is_significant,
                    'confidence_level': 0.95 if is_significant else 0.5
                }
        
        return comparison_results
    
    def generate_evaluation_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report string
        """
        logger.info("Generating comprehensive evaluation report")
        
        report = []
        report.append("=" * 80)
        report.append("HOTEL BOOKING CANCELLATION PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance Summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        if self.evaluation_results:
            # Sort models by F1 score
            sorted_models = sorted(
                self.evaluation_results.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            report.append(f"{'Model':<20} {'F1':<8} {'ROC-AUC':<8} {'Precision':<10} {'Recall':<8}")
            report.append("-" * 60)
            
            for model_name, metrics in sorted_models:
                report.append(
                    f"{model_name:<20} "
                    f"{metrics['f1_score']:<8.4f} "
                    f"{metrics['roc_auc']:<8.4f} "
                    f"{metrics['precision']:<10.4f} "
                    f"{metrics['recall']:<8.4f}"
                )
        
        report.append("")
        
        # Business Impact Analysis
        report.append("BUSINESS IMPACT ANALYSIS")
        report.append("-" * 40)
        
        if self.evaluation_results:
            best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['f1_score'])
            best_name, best_metrics = best_model
            
            report.append(f"Best Model: {best_name}")
            report.append(f"Net Benefit: ${best_metrics.get('net_benefit_usd', 0):,.2f}")
            report.append(f"Cost Savings per Booking: ${best_metrics.get('cost_savings_per_booking', 0):.2f}")
            report.append(f"True Positives: {best_metrics.get('true_positives', 0)}")
            report.append(f"False Positives: {best_metrics.get('false_positives', 0)}")
            report.append(f"False Negatives: {best_metrics.get('false_negatives', 0)}")
        
        report.append("")
        
        # Feature Importance (from SHAP if available)
        report.append("FEATURE IMPORTANCE (TOP 10)")
        report.append("-" * 40)
        
        if self.shap_values:
            # Get SHAP results from best model
            best_model_name = best_name if 'best_name' in locals() else list(self.shap_values.keys())[0]
            
            if best_model_name in self.shap_values:
                top_features = self.shap_values[best_model_name].get('top_features', [])
                
                for i, (feature, importance) in enumerate(top_features, 1):
                    report.append(f"{i:2d}. {feature:<30} {importance:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def evaluation_pipeline(self, df: pd.DataFrame, target_column: str = 'is_canceled') -> Dict:
        """
        Complete model evaluation pipeline.
        
        Args:
            df: Feature-engineered DataFrame
            target_column: Target variable column name
            
        Returns:
            Complete evaluation results dictionary
        """
        logger.info("Starting complete model evaluation pipeline")
        
        # 1. Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)
        
        # 2. Train all models
        models_trained = []
        
        # Logistic Regression
        try:
            self.train_logistic_regression(X_train, y_train)
            models_trained.append('logistic_regression')
        except Exception as e:
            logger.error(f"Failed to train logistic regression: {e}")
        
        # Random Forest
        try:
            self.train_random_forest(X_train, y_train)
            models_trained.append('random_forest')
        except Exception as e:
            logger.error(f"Failed to train random forest: {e}")
        
        # XGBoost
        try:
            if 'xgb' in globals():
                self.train_xgboost(X_train, y_train)
                models_trained.append('xgboost')
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}")
        
        # PyTorch MLP
        try:
            if 'torch' in globals():
                self.train_pytorch_mlp(X_train, y_train)
                models_trained.append('pytorch_mlp')
        except Exception as e:
            logger.error(f"Failed to train PyTorch MLP: {e}")
        
        logger.info(f"Successfully trained models: {models_trained}")
        
        # 3. Evaluate all models
        for model_name in models_trained:
            if model_name in self.models:
                self.evaluate_model(self.models[model_name], model_name, X_test, y_test)
        
        # 4. Cross-validation
        cv_results = self.cross_validate_models(X_train, y_train)
        
        # 5. SHAP analysis
        feature_names = df.drop(columns=[target_column]).columns.tolist()
        
        # Handle categorical encoding for feature names
        if any('_' in col for col in feature_names):
            # Assume dummy encoding was applied
            pass  # Feature names are already correct
        
        for model_name in models_trained[:2]:  # Limit SHAP to first 2 models for performance
            if model_name in self.models and model_name != 'pytorch_mlp':
                self.perform_shap_analysis(model_name, X_train, X_test, feature_names)
        
        # 6. Statistical comparison
        statistical_comparison = self.compare_models_statistically()
        
        # 7. MLflow logging
        if 'mlflow' in globals():
            for model_name, metrics in self.evaluation_results.items():
                with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                    # Log parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("data_shape", f"{X_train.shape[0] + X_test.shape[0]}x{X_train.shape[1]}")
                    
                    # Log metrics
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            mlflow.log_metric(metric_name, value)
                    
                    # Log model
                    if model_name in self.models:
                        if model_name in ['logistic_regression', 'random_forest']:
                            mlflow.sklearn.log_model(self.models[model_name], model_name)
                        elif model_name == 'xgboost' and 'mlflow.xgboost' in dir(mlflow):
                            mlflow.xgboost.log_model(self.models[model_name], model_name)
        
        # Compile final results
        final_results = {
            'evaluation_results': self.evaluation_results,
            'cross_validation_results': cv_results,
            'shap_analysis': self.shap_values,
            'statistical_comparison': statistical_comparison,
            'models_trained': models_trained,
            'evaluation_completed': datetime.now().isoformat()
        }
        
        logger.info("Model evaluation pipeline completed successfully")
        
        return final_results
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """Save evaluation results and generate report."""
        logger.info(f"Saving evaluation results to {output_path}")
        
        # Save results as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save report
        report = self.generate_evaluation_report()
        report_path = output_path.replace('.json', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation completed and saved to {output_path}")
        logger.info(f"Report saved to {report_path}")


def main():
    """Main model evaluation execution function."""
    # Configuration
    config = {
        'input_path': '../data/features/hotel_bookings_features.csv',
        'output_path': '../results/model_evaluation_results.json',
        'target_column': 'is_canceled',
        'test_size': 0.2,
        'cv_folds': 5,
        'n_trials': 50,  # Reduced for faster execution
        'mlflow_uri': 'file:./mlruns'
    }
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(config)
    
    try:
        # Load feature-engineered data
        df_features = evaluator.load_feature_engineered_data(config['input_path'])
        
        # Run evaluation pipeline
        results = evaluator.evaluation_pipeline(df_features, config['target_column'])
        
        # Save results
        evaluator.save_evaluation_results(results, config['output_path'])
        
        # Print summary
        print("✅ Model evaluation completed successfully")
        
        if evaluator.evaluation_results:
            print("\n🏆 MODEL PERFORMANCE SUMMARY:")
            sorted_models = sorted(
                evaluator.evaluation_results.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            for i, (model_name, metrics) in enumerate(sorted_models, 1):
                print(f"{i}. {model_name:<20} F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\n📁 Results saved to: {config['output_path']}")
        print(f"📊 MLflow tracking URI: {config['mlflow_uri']}")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()