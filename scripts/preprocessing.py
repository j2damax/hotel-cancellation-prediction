import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

class HotelDataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.preprocessing_steps = []
        
    def load_data(self, file_path='data/raw/hotel_booking.csv'):
        """Load raw hotel booking data"""
        df = pd.read_csv(file_path)
        self.logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        self.preprocessing_steps.append(f"Data loaded: {df.shape}")
        return df
    
    def assess_data_quality(self, df):
        """Assess data quality - missing values and business logic checks"""
        self.logger.info("Assessing data quality...")
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                self.logger.info(f"Missing: {col} - {count:,} ({pct:.1f}%)")
        
        self.preprocessing_steps.append("Data quality assessed")
        return {'missing_values': missing_summary.to_dict()}
    
    def handle_missing_values(self, df):
        """Handle missing values with simple imputation strategy"""
        self.logger.info("Handling missing values...")
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        # Drop high missing columns (>50%)
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() / len(df_clean) > 0.5:
                df_clean = df_clean.drop(columns=[col])
                self.logger.info(f"Dropped {col} (>50% missing)")
        
        # Simple imputation
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
        
        missing_after = df_clean.isnull().sum().sum()
        self.logger.info(f"Missing values: {missing_before} -> {missing_after}")
        self.preprocessing_steps.append(f"Missing values handled: {missing_before} -> {missing_after}")
        return df_clean
    
    def detect_and_handle_outliers(self, df):
        """Detect and handle outliers using IQR method with capping"""
        self.logger.info("Handling outliers...")
        df_clean = df.copy()
        numerical_cols = ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
        existing_cols = [col for col in numerical_cols if col in df_clean.columns]
        
        total_capped = 0
        for col in existing_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            if outliers > 0:
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                total_capped += outliers
                self.logger.info(f"Capped {outliers} outliers in {col}")
        
        self.logger.info(f"Total outliers capped: {total_capped}")
        self.preprocessing_steps.append(f"Outliers capped: {total_capped}")
        return df_clean
    
    def prepare_for_modeling(self, df):
        """Final preparation: remove leakage, encode categories"""
        self.logger.info("Preparing for modeling...")
        df_final = df.copy()
        
        # Remove leakage columns
        leakage_cols = ['reservation_status', 'reservation_status_date']
        existing_leakage = [col for col in leakage_cols if col in df_final.columns]
        if existing_leakage:
            df_final = df_final.drop(columns=existing_leakage)
            self.logger.info(f"Removed leakage columns: {existing_leakage}")
        
        # Remove low-value columns
        low_value_cols = ['company', 'agent']
        existing_low = [col for col in low_value_cols if col in df_final.columns]
        if existing_low:
            df_final = df_final.drop(columns=existing_low)
            self.logger.info(f"Removed low-value columns: {existing_low}")
        
        # Encode categorical variables
        categorical_cols = df_final.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'is_canceled']
        
        encoded_count = 0
        for col in categorical_cols:
            if df_final[col].nunique() <= 20:  # Only encode low-cardinality categoricals
                le = LabelEncoder()
                df_final[col] = le.fit_transform(df_final[col].astype(str))
                encoded_count += 1
        
        self.logger.info(f"Encoded {encoded_count} categorical columns")
        self.preprocessing_steps.append(f"Data prepared: {encoded_count} columns encoded")
        return df_final
    
    def add_basic_derived_features(self, df):
        """Add only essential derived features needed for basic analysis"""
        self.logger.info("Adding basic derived features...")
        df_features = df.copy()
        features_added = []
        
        # Total stay duration - fundamental business metric
        if all(col in df_features.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            df_features['total_stay_duration'] = (
                df_features['stays_in_weekend_nights'] + df_features['stays_in_week_nights']
            )
            features_added.append('total_stay_duration')
        
        # Total guests - basic guest count
        if all(col in df_features.columns for col in ['adults', 'children', 'babies']):
            df_features['total_guests'] = (
                df_features['adults'] + df_features['children'] + df_features['babies']
            )
            features_added.append('total_guests')
        
        self.logger.info(f"Added {len(features_added)} basic features: {features_added}")
        self.preprocessing_steps.append(f"Basic derived features: {len(features_added)} features added")
        return df_features
    
    def get_preprocessing_report(self):
        """Generate preprocessing report"""
        report = []
        report.append("PREPROCESSING REPORT - ESSENTIAL DATA CLEANING")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("Scope: Data cleaning, outlier handling, basic encoding")
        report.append("Note: Advanced feature engineering moved to separate phase")
        report.append("")
        
        for i, step in enumerate(self.preprocessing_steps, 1):
            report.append(f"{i}. {step}")
        
        return "\n".join(report)
