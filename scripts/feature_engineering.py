"""
Feature engineering pipeline for hotel booking cancellation prediction.
Implements advanced feature creation strategies including categorical encoding,
temporal features, and domain-specific business features.

Academic Research Framework - NIB 7072 Coursework
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
import calendar
from typing import Tuple, Dict, List, Optional

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold

# Advanced feature engineering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for hotel booking data.
    
    Implements strategies from features.md including:
    - Advanced categorical encoding (mean target encoding)
    - Temporal and seasonal features
    - Guest behavior and composition features
    - Financial and revenue optimization features
    - Risk assessment features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary for feature engineering parameters
        """
        self.config = config or {}
        self.encoding_maps = {}
        self.feature_importance = {}
        self.engineering_log = {}
        
    def load_preprocessed_data(self, file_path: str) -> pd.DataFrame:
        """Load preprocessed hotel booking data."""
        logger.info(f"Loading preprocessed data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Preprocessed data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Preprocessed data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
    def implement_mean_target_encoding(self, df: pd.DataFrame, 
                                     categorical_columns: Optional[List[str]] = None,
                                     target_column: str = 'is_canceled',
                                     smoothing_factor: int = 10,
                                     cv_folds: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        Implement mean target encoding with cross-validation to prevent overfitting.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical columns to encode
            target_column: Target variable column name
            smoothing_factor: Smoothing parameter for regularization
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with encoded features and encoding results dictionary
        """
        logger.info("Implementing mean target encoding with cross-validation")
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
        
        df_encoded = df.copy()
        encoding_results = {}
        global_mean = df[target_column].mean()
        
        # Set up cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in categorical_columns:
            if col in df.columns:
                logger.info(f"Encoding {col} with mean target encoding")
                
                # Analyze encoding potential
                category_stats = df.groupby(col)[target_column].agg(['mean', 'count', 'std']).reset_index()
                category_stats.columns = [col, 'target_mean', 'count', 'std']
                
                # Calculate effectiveness metrics
                overall_variance = global_mean * (1 - global_mean)
                category_variance = np.var(category_stats['target_mean'])
                effectiveness_ratio = category_variance / overall_variance if overall_variance > 0 else 0
                
                # Implement cross-validation encoding
                encoded_values = np.zeros(len(df))
                
                for train_idx, val_idx in kf.split(df):
                    train_data = df.iloc[train_idx]
                    val_data = df.iloc[val_idx]
                    
                    # Calculate encoding map from training data only
                    encoding_map = train_data.groupby(col)[target_column].agg(['mean', 'count'])
                    
                    # Apply smoothing
                    encoding_map['smoothed_mean'] = (
                        (encoding_map['count'] * encoding_map['mean'] + smoothing_factor * global_mean) /
                        (encoding_map['count'] + smoothing_factor)
                    )
                    
                    # Map validation data
                    val_encoded = val_data[col].map(encoding_map['smoothed_mean'])
                    val_encoded = val_encoded.fillna(global_mean)  # Handle unseen categories
                    
                    encoded_values[val_idx] = val_encoded
                
                # Store encoded column
                encoded_col_name = f'{col}_target_encoded'
                df_encoded[encoded_col_name] = encoded_values
                
                # Calculate correlation with target
                correlation_with_target = df_encoded[encoded_col_name].corr(df_encoded[target_column])
                
                # Store encoding results
                encoding_results[col] = {
                    'encoded_column': encoded_col_name,
                    'effectiveness_ratio': effectiveness_ratio,
                    'correlation_with_target': correlation_with_target,
                    'unique_categories': len(category_stats),
                    'target_range': category_stats['target_mean'].max() - category_stats['target_mean'].min()
                }
        
        self.encoding_maps = encoding_results
        logger.info(f"Mean target encoding completed for {len(encoding_results)} columns")
        
        return df_encoded, encoding_results
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal and seasonal features.
        """
        logger.info("Creating temporal and seasonal features")
        
        df_temporal = df.copy()
        
        # Season mapping based on arrival month
        if 'arrival_date_month' in df.columns:
            season_map = {
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            }
            df_temporal['season'] = df['arrival_date_month'].map(season_map)
            
            # Peak season indicator (for tourism)
            peak_months = [6, 7, 8, 12]  # Summer and December
            df_temporal['is_peak_season'] = df['arrival_date_month'].isin(peak_months).astype(int)
            
            # Month cyclical encoding
            df_temporal['month_sin'] = np.sin(2 * np.pi * df['arrival_date_month'] / 12)
            df_temporal['month_cos'] = np.cos(2 * np.pi * df['arrival_date_month'] / 12)
        
        # Weekend arrival indicator
        if 'arrival_date_week_number' in df.columns:
            # Assuming week starts Monday (0), weekend is 5-6
            df_temporal['is_weekend_arrival'] = (df['arrival_date_week_number'] % 7).isin([5, 6]).astype(int)
        
        # Lead time categories
        if 'lead_time' in df.columns:
            df_temporal['lead_time_category'] = pd.cut(
                df['lead_time'],
                bins=[-1, 7, 30, 90, 180, 365, float('inf')],
                labels=['Last_Minute', 'Short_Term', 'Medium_Term', 'Long_Term', 'Very_Long_Term', 'Extreme']
            )
            
            # Lead time risk indicator
            df_temporal['high_lead_time_risk'] = (df['lead_time'] > 180).astype(int)
        
        logger.info(f"Created {len([c for c in df_temporal.columns if c not in df.columns])} temporal features")
        
        return df_temporal
    
    def create_guest_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create guest behavior and composition features.
        """
        logger.info("Creating guest behavior and composition features")
        
        df_guest = df.copy()
        
        # Basic guest composition
        if all(col in df.columns for col in ['adults', 'children', 'babies']):
            df_guest['total_guests'] = df['adults'] + df['children'] + df['babies']
            df_guest['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
            df_guest['adults_to_children_ratio'] = df['adults'] / (df['children'] + df['babies'] + 1)
            
            # Guest type segmentation
            def categorize_guest_type(row):
                if row['children'] > 0 or row['babies'] > 0:
                    return 'Family'
                elif row['adults'] == 1:
                    return 'Solo'
                elif row['adults'] == 2:
                    return 'Couple'
                else:
                    return 'Group'
            
            df_guest['guest_type'] = df.apply(categorize_guest_type, axis=1)
        
        # Stay duration features
        if all(col in df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            df_guest['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            df_guest['weekend_to_weekday_ratio'] = (
                df['stays_in_weekend_nights'] / (df['stays_in_week_nights'] + 1)
            )
            
            # Stay duration categories
            df_guest['stay_duration_category'] = pd.cut(
                df_guest['total_stay_nights'],
                bins=[0, 1, 3, 7, 14, float('inf')],
                labels=['Day_Trip', 'Short_Stay', 'Weekend', 'Week', 'Extended']
            )
        
        # Previous booking behavior
        if 'previous_cancellations' in df.columns:
            df_guest['has_previous_cancellations'] = (df['previous_cancellations'] > 0).astype(int)
            df_guest['cancellation_risk_score'] = np.log1p(df['previous_cancellations'])
        
        if 'previous_bookings_not_canceled' in df.columns:
            df_guest['is_loyal_customer'] = (df['previous_bookings_not_canceled'] > 2).astype(int)
            df_guest['loyalty_score'] = np.log1p(df['previous_bookings_not_canceled'])
        
        # Special requests behavior
        if 'total_of_special_requests' in df.columns:
            df_guest['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
            df_guest['special_requests_per_guest'] = (
                df['total_of_special_requests'] / df_guest.get('total_guests', 1)
            )
        
        logger.info(f"Created {len([c for c in df_guest.columns if c not in df.columns])} guest behavior features")
        
        return df_guest
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial and revenue optimization features.
        """
        logger.info("Creating financial and revenue optimization features")
        
        df_financial = df.copy()
        
        # Revenue calculations
        if 'adr' in df.columns and 'total_stay_nights' in df.columns:
            df_financial['total_revenue'] = df['adr'] * df['total_stay_nights']
            df_financial['revenue_per_guest'] = (
                df_financial['total_revenue'] / df_financial.get('total_guests', 1)
            )
        
        # ADR categories and percentiles
        if 'adr' in df.columns:
            df_financial['adr_percentile'] = df['adr'].rank(pct=True)
            
            adr_percentiles = df['adr'].quantile([0.25, 0.5, 0.75]).values
            df_financial['adr_category'] = pd.cut(
                df['adr'],
                bins=[0] + list(adr_percentiles) + [float('inf')],
                labels=['Budget', 'Economy', 'Standard', 'Premium']
            )
            
            # Price deviation from market segment mean
            if 'market_segment' in df.columns:
                segment_mean_adr = df.groupby('market_segment')['adr'].transform('mean')
                df_financial['adr_deviation_from_segment'] = df['adr'] - segment_mean_adr
                df_financial['adr_premium_ratio'] = df['adr'] / segment_mean_adr
        
        # Deposit and payment features
        if 'deposit_type' in df.columns:
            df_financial['has_deposit'] = (df['deposit_type'] != 'No Deposit').astype(int)
        
        # Booking changes cost indicator
        if 'booking_changes' in df.columns:
            df_financial['has_booking_changes'] = (df['booking_changes'] > 0).astype(int)
            df_financial['booking_stability_score'] = 1 / (1 + df['booking_changes'])
        
        logger.info(f"Created {len([c for c in df_financial.columns if c not in df.columns])} financial features")
        
        return df_financial
    
    def create_risk_assessment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk assessment and booking pattern features.
        """
        logger.info("Creating risk assessment features")
        
        df_risk = df.copy()
        
        # Comprehensive risk score
        risk_factors = []
        
        if 'lead_time' in df.columns:
            # High lead time increases risk
            lead_time_risk = (df['lead_time'] > df['lead_time'].quantile(0.75)).astype(int)
            risk_factors.append(lead_time_risk)
        
        if 'has_previous_cancellations' in df.columns:
            risk_factors.append(df['has_previous_cancellations'])
        
        if 'has_booking_changes' in df.columns:
            risk_factors.append(df['has_booking_changes'])
        
        if 'market_segment' in df.columns:
            # Online TA typically higher risk
            online_ta_risk = (df['market_segment'] == 'Online TA').astype(int)
            risk_factors.append(online_ta_risk)
        
        # Combine risk factors
        if risk_factors:
            df_risk['composite_risk_score'] = sum(risk_factors) / len(risk_factors)
            df_risk['high_risk_booking'] = (df_risk['composite_risk_score'] > 0.5).astype(int)
        
        # Market volatility indicators
        if 'customer_type' in df.columns:
            df_risk['is_transient_customer'] = (df['customer_type'] == 'Transient').astype(int)
        
        # Room assignment risk
        if all(col in df.columns for col in ['reserved_room_type', 'assigned_room_type']):
            df_risk['room_type_changed'] = (
                df['reserved_room_type'] != df['assigned_room_type']
            ).astype(int)
        
        logger.info(f"Created {len([c for c in df_risk.columns if c not in df.columns])} risk assessment features")
        
        return df_risk
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction and polynomial features.
        """
        logger.info("Creating interaction features")
        
        df_interaction = df.copy()
        
        # Key feature interactions based on domain knowledge
        interactions_to_create = [
            ('lead_time', 'adr'),  # Lead time vs price interaction
            ('total_guests', 'total_stay_nights'),  # Group size vs stay duration
            ('adr', 'total_stay_nights'),  # Price vs duration
            ('is_family', 'season'),  # Family bookings seasonal patterns
            ('market_segment', 'lead_time'),  # Segment vs planning behavior
        ]
        
        created_interactions = 0
        for feat1, feat2 in interactions_to_create:
            if created_interactions >= max_interactions:
                break
                
            if feat1 in df.columns and feat2 in df.columns:
                # Numerical interactions
                if (df[feat1].dtype in ['int64', 'float64'] and 
                    df[feat2].dtype in ['int64', 'float64']):
                    
                    interaction_name = f'{feat1}_x_{feat2}'
                    df_interaction[interaction_name] = df[feat1] * df[feat2]
                    created_interactions += 1
        
        # Polynomial features for key numerical variables
        key_numerical_features = ['lead_time', 'adr', 'total_guests']
        existing_features = [f for f in key_numerical_features if f in df.columns]
        
        if len(existing_features) >= 2:
            # Create squared terms for key features
            for feat in existing_features[:3]:  # Limit to avoid explosion
                df_interaction[f'{feat}_squared'] = df[feat] ** 2
                created_interactions += 1
                
                if created_interactions >= max_interactions:
                    break
        
        logger.info(f"Created {created_interactions} interaction features")
        
        return df_interaction
    
    def calculate_feature_importance(self, df: pd.DataFrame, 
                                   target_column: str = 'is_canceled') -> Dict:
        """
        Calculate feature importance using multiple methods.
        """
        logger.info("Calculating feature importance")
        
        # Prepare features
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        le = LabelEncoder()
        for col in categorical_cols:
            X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        importance_results = {}
        
        # Correlation-based importance
        correlations = {}
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            corr = abs(X_encoded[col].corr(y))
            correlations[col] = corr
        
        importance_results['correlation'] = correlations
        
        # Mutual information importance
        try:
            mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
            mi_importance = dict(zip(X_encoded.columns, mi_scores))
            importance_results['mutual_info'] = mi_importance
        except Exception as e:
            logger.warning(f"Could not calculate mutual information: {e}")
        
        # Statistical F-test importance
        try:
            f_scores, p_values = f_classif(X_encoded, y)
            f_importance = dict(zip(X_encoded.columns, f_scores))
            importance_results['f_test'] = f_importance
        except Exception as e:
            logger.warning(f"Could not calculate F-test scores: {e}")
        
        self.feature_importance = importance_results
        
        return importance_results
    
    def select_optimal_features(self, df: pd.DataFrame, 
                              target_column: str = 'is_canceled',
                              max_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select optimal feature set based on importance scores.
        """
        logger.info(f"Selecting optimal features (max: {max_features})")
        
        importance_scores = self.calculate_feature_importance(df, target_column)
        
        # Combine importance scores (weighted average)
        combined_scores = {}
        feature_names = list(df.drop(columns=[target_column]).columns)
        
        for feature in feature_names:
            score = 0
            weight_sum = 0
            
            if 'correlation' in importance_scores and feature in importance_scores['correlation']:
                score += importance_scores['correlation'][feature] * 0.3
                weight_sum += 0.3
            
            if 'mutual_info' in importance_scores and feature in importance_scores['mutual_info']:
                score += importance_scores['mutual_info'][feature] * 0.4
                weight_sum += 0.4
            
            if 'f_test' in importance_scores and feature in importance_scores['f_test']:
                normalized_f = importance_scores['f_test'][feature] / max(importance_scores['f_test'].values())
                score += normalized_f * 0.3
                weight_sum += 0.3
            
            if weight_sum > 0:
                combined_scores[feature] = score / weight_sum
            else:
                combined_scores[feature] = 0
        
        # Select top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:max_features]]
        
        # Always include target column
        final_columns = selected_features + [target_column]
        df_selected = df[final_columns]
        
        logger.info(f"Selected {len(selected_features)} optimal features")
        
        return df_selected, selected_features
    
    def feature_engineering_pipeline(self, df: pd.DataFrame, 
                                   target_column: str = 'is_canceled',
                                   max_features: int = 50) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Preprocessed input DataFrame
            target_column: Target variable column name
            max_features: Maximum number of features to select
            
        Returns:
            Feature-engineered DataFrame ready for model training
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # 1. Advanced categorical encoding
        df_encoded, encoding_results = self.implement_mean_target_encoding(df, target_column=target_column)
        
        # 2. Temporal feature engineering
        df_temporal = self.create_temporal_features(df_encoded)
        
        # 3. Guest behavior features
        df_guest = self.create_guest_behavior_features(df_temporal)
        
        # 4. Financial features
        df_financial = self.create_financial_features(df_guest)
        
        # 5. Risk assessment features
        df_risk = self.create_risk_assessment_features(df_financial)
        
        # 6. Interaction features
        df_interaction = self.create_interaction_features(df_risk)
        
        # 7. Feature selection
        df_final, selected_features = self.select_optimal_features(
            df_interaction, target_column, max_features
        )
        
        # Store engineering metadata
        self.engineering_log = {
            'original_features': len(df.columns),
            'engineered_features': len(df_interaction.columns),
            'selected_features': len(selected_features),
            'encoding_results': encoding_results,
            'feature_importance': self.feature_importance,
            'selected_feature_list': selected_features,
            'engineering_completed': datetime.now().isoformat()
        }
        
        logger.info(f"Feature engineering pipeline completed: {df.shape} -> {df_final.shape}")
        
        return df_final
    
    def save_engineered_data(self, df: pd.DataFrame, output_path: str):
        """Save feature-engineered data and metadata."""
        logger.info(f"Saving feature-engineered data to {output_path}")
        
        df.to_csv(output_path, index=False)
        
        # Save engineering metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.engineering_log, f, indent=2, default=str)
        
        logger.info(f"Feature engineering completed and saved to {output_path}")


def main():
    """Main feature engineering execution function."""
    # Configuration
    config = {
        'input_path': '../data/processed/hotel_bookings_preprocessed.csv',
        'output_path': '../data/features/hotel_bookings_features.csv',
        'target_column': 'is_canceled',
        'max_features': 50,
        'cv_folds': 5,
        'smoothing_factor': 10
    }
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    try:
        # Load preprocessed data
        df_processed = feature_engineer.load_preprocessed_data(config['input_path'])
        
        # Run feature engineering pipeline
        df_features = feature_engineer.feature_engineering_pipeline(
            df_processed,
            target_column=config['target_column'],
            max_features=config['max_features']
        )
        
        # Save results
        feature_engineer.save_engineered_data(df_features, config['output_path'])
        
        print("✅ Feature engineering completed successfully")
        print(f"📊 Final dataset shape: {df_features.shape}")
        print(f"📁 Saved to: {config['output_path']}")
        
        # Display feature importance summary
        if feature_engineer.feature_importance:
            print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
            correlation_scores = feature_engineer.feature_importance.get('correlation', {})
            top_features = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature:<30} | Correlation: {score:.4f}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()