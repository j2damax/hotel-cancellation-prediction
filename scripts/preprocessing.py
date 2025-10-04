"""
Data preprocessing pipeline for hotel booking cancellation prediction.
Implements comprehensive preprocessing strategies including missing value handling,
outlier treatment, and class imbalance correction.

Academic Research Framework - NIB 7072 Coursework
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

# Statistical analysis
from scipy import stats
from scipy.stats import zscore, iqr

# Preprocessing tools
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for hotel booking data.
    
    Implements strategies from preprocessing.md including:
    - Missing value analysis and imputation
    - Business logic validation  
    - Strategic column dropping
    - Outlier detection and treatment
    - Class imbalance handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or {}
        self.scaler = None
        self.imputers = {}
        self.dropped_columns = []
        self.preprocessing_log = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw hotel booking data."""
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive missing value analysis.
        
        Returns:
            Dictionary with missing value statistics and patterns
        """
        logger.info("Analyzing missing value patterns")
        
        missing_summary = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        }
        
        # High missing value columns (>70%)
        high_missing = df.isnull().sum() / len(df) > 0.7
        missing_summary['high_missing_columns'] = df.columns[high_missing].tolist()
        
        return missing_summary
    
    def validate_business_logic(self, df: pd.DataFrame) -> Dict:
        """
        Validate business logic and data consistency.
        Implements validation functions from preprocessing.md
        """
        logger.info("Validating business logic")
        
        validation_results = {}
        
        # Check for impossible guest combinations
        if all(col in df.columns for col in ['adults', 'children', 'babies']):
            no_guests = df[(df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)]
            validation_results['no_guests'] = len(no_guests)
        
        # Check for negative values in count fields
        count_fields = ['adults', 'children', 'babies', 'stays_in_weekend_nights', 'stays_in_week_nights']
        for field in count_fields:
            if field in df.columns:
                negative_count = (df[field] < 0).sum()
                validation_results[f'negative_{field}'] = negative_count
        
        # Reservation status consistency validation
        if 'reservation_status' in df.columns and 'is_canceled' in df.columns:
            status_vs_cancel = pd.crosstab(df['reservation_status'], df['is_canceled'])
            validation_results['status_consistency'] = status_vs_cancel.to_dict()
        
        return validation_results
    
    def implement_column_dropping_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Strategic column dropping based on business logic and data quality.
        Implements pipeline from preprocessing.md
        """
        logger.info("Implementing strategic column dropping pipeline")
        
        columns_to_drop = []
        dropping_rationale = {}
        
        # High missing value columns (>70%)
        high_missing_threshold = 0.7
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > high_missing_threshold:
                columns_to_drop.append(col)
                dropping_rationale[col] = f"High missing values ({missing_pct:.1%})"
        
        # Data leakage risk columns
        leakage_columns = ['reservation_status', 'reservation_status_date']
        for col in leakage_columns:
            if col in df.columns:
                columns_to_drop.append(col)
                dropping_rationale[col] = "Data leakage risk"
        
        # High cardinality columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() > 100:
                columns_to_drop.append(col)
                dropping_rationale[col] = f"High cardinality ({df[col].nunique()} unique values)"
        
        # Remove duplicates and drop columns
        columns_to_drop = list(set(columns_to_drop))
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
        
        self.dropped_columns = columns_to_drop
        self.preprocessing_log['column_dropping'] = dropping_rationale
        
        logger.info(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
        
        return df_cleaned, columns_to_drop
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategic missing value imputation based on business context.
        """
        logger.info("Handling missing values with domain-specific strategies")
        
        df_imputed = df.copy()
        
        # Business logic imputation
        if 'children' in df_imputed.columns:
            df_imputed['children'].fillna(0, inplace=True)
        
        if 'babies' in df_imputed.columns:
            df_imputed['babies'].fillna(0, inplace=True)
        
        # Numerical features - use median imputation
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_imputed[col].isnull().any():
                median_value = df_imputed[col].median()
                df_imputed[col].fillna(median_value, inplace=True)
        
        # Categorical features - use mode imputation
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_imputed[col].isnull().any():
                mode_value = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else 'Unknown'
                df_imputed[col].fillna(mode_value, inplace=True)
        
        return df_imputed
    
    def detect_and_treat_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Multi-method outlier detection and treatment.
        """
        logger.info(f"Detecting and treating outliers using {method} method")
        
        df_treated = df.copy()
        numerical_cols = df_treated.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop('is_canceled', errors='ignore')
        
        outlier_info = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df_treated[col].quantile(0.25)
                Q3 = df_treated[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)
                outlier_count = outliers.sum()
                
                # Cap outliers instead of removing
                df_treated[col] = np.clip(df_treated[col], lower_bound, upper_bound)
                
                outlier_info[col] = {
                    'method': 'IQR',
                    'outlier_count': outlier_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_treated[col]))
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                # Cap outliers at 3 standard deviations
                mean_val = df_treated[col].mean()
                std_val = df_treated[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                df_treated[col] = np.clip(df_treated[col], lower_bound, upper_bound)
                
                outlier_info[col] = {
                    'method': 'Z-Score',
                    'outlier_count': outlier_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        self.preprocessing_log['outlier_treatment'] = outlier_info
        logger.info(f"Outlier treatment completed for {len(numerical_cols)} numerical columns")
        
        return df_treated
    
    def apply_resampling(self, df: pd.DataFrame, target_column: str = 'is_canceled', 
                        method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class imbalance handling techniques.
        """
        logger.info(f"Applying {method} resampling technique")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Ensure all features are numerical for resampling
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.warning(f"Categorical columns found: {list(categorical_cols)}. Encoding required.")
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=42)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled
        
        logger.info(f"Resampling completed: {df.shape} -> {df_resampled.shape}")
        
        return df_resampled, y_resampled
    
    def fit_scaler(self, df: pd.DataFrame, target_column: str = 'is_canceled') -> StandardScaler:
        """Fit scaler on numerical features."""
        logger.info("Fitting scaler on numerical features")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop(target_column, errors='ignore')
        
        self.scaler = StandardScaler()
        self.scaler.fit(df[numerical_cols])
        
        return self.scaler
    
    def transform_data(self, df: pd.DataFrame, target_column: str = 'is_canceled') -> pd.DataFrame:
        """Apply scaling transformation to numerical features."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        df_scaled = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop(target_column, errors='ignore')
        
        df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_scaled
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str = 'is_canceled', 
                           apply_resampling: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            target_column: Target variable column name
            apply_resampling: Whether to apply class imbalance handling
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # 1. Data quality assessment
        missing_analysis = self.analyze_missing_values(df)
        validation_results = self.validate_business_logic(df)
        
        # 2. Strategic column dropping
        df_cleaned, dropped_cols = self.implement_column_dropping_pipeline(df)
        
        # 3. Missing value handling
        df_imputed = self.handle_missing_values(df_cleaned)
        
        # 4. Outlier treatment
        df_outliers_treated = self.detect_and_treat_outliers(df_imputed)
        
        # 5. Class imbalance handling (optional)
        if apply_resampling and target_column in df_outliers_treated.columns:
            df_resampled, _ = self.apply_resampling(df_outliers_treated, target_column)
            final_df = df_resampled
        else:
            final_df = df_outliers_treated
        
        # 6. Feature scaling (fit only, transform separately)
        self.fit_scaler(final_df, target_column)
        
        # Store preprocessing metadata
        self.preprocessing_log.update({
            'missing_analysis': missing_analysis,
            'validation_results': validation_results,
            'final_shape': final_df.shape,
            'preprocessing_completed': datetime.now().isoformat()
        })
        
        logger.info(f"Preprocessing pipeline completed: {df.shape} -> {final_df.shape}")
        
        return final_df
    
    def save_preprocessed_data(self, df: pd.DataFrame, output_path: str):
        """Save preprocessed data and metadata."""
        logger.info(f"Saving preprocessed data to {output_path}")
        
        df.to_csv(output_path, index=False)
        
        # Save preprocessing metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.preprocessing_log, f, indent=2, default=str)
        
        logger.info(f"Preprocessing completed and saved to {output_path}")


def main():
    """Main preprocessing execution function."""
    # Configuration
    config = {
        'input_path': '../data/raw/hotel_bookings.csv',
        'output_path': '../data/processed/hotel_bookings_preprocessed.csv',
        'target_column': 'is_canceled',
        'apply_resampling': True,
        'outlier_method': 'iqr'
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    try:
        # Load data
        df_raw = preprocessor.load_data(config['input_path'])
        
        # Run preprocessing pipeline
        df_processed = preprocessor.preprocess_pipeline(
            df_raw, 
            target_column=config['target_column'],
            apply_resampling=config['apply_resampling']
        )
        
        # Save results
        preprocessor.save_preprocessed_data(df_processed, config['output_path'])
        
        print("✅ Preprocessing completed successfully")
        print(f"📊 Final dataset shape: {df_processed.shape}")
        print(f"📁 Saved to: {config['output_path']}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()