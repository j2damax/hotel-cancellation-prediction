# Data Preprocessing Instructions - Essential Requirements Only
## Hotel Booking Cancellation Prediction - NIB 7072 Coursework

**Objective**: Create a clean, preprocessed dataset ready for model training with minimal essential steps based on EDA findings.

---

## 🎯 Essential 5-Phase Preprocessing Pipeline

### Phase 1: Environment Setup & Data Loading

**Required Libraries:**
```python
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
```

**Load Data:**
```python
def load_data(file_path='data/raw/hotel_booking.csv'):
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    print(f"📊 Cancellation rate: {df['is_canceled'].mean():.3f}")
    return df
```

---

### Phase 2: Data Quality Assessment (Essential Checks Only)

**Missing Value Analysis:**
```python
def assess_missing_values(df):
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    
    print("📋 Missing Value Summary:")
    missing_df = pd.DataFrame({
        'Missing_Count': missing_summary,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    return missing_df
```

**Business Logic Validation:**
```python
def validate_business_logic(df):
    issues = []
    
    # Check impossible guest combinations
    no_guests = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)
    if no_guests.sum() > 0:
        issues.append(f"No guests bookings: {no_guests.sum()}")
    
    # Check data leakage
    if 'reservation_status' in df.columns:
        leakage_rate = (df[df['is_canceled']==1]['reservation_status'] == 'Canceled').mean()
        if leakage_rate > 0.8:
            issues.append(f"⚠️ Data leakage risk: {leakage_rate:.1%}")
    
    return issues
```

---

### Phase 3: Missing Value Treatment (Simple Strategy)

```python
def handle_missing_values(df):
    df_clean = df.copy()
    
    # Strategy 1: Drop high missing columns (>50% missing)
    high_missing = df_clean.isnull().sum() / len(df_clean) > 0.5
    cols_to_drop = df_clean.columns[high_missing].tolist()
    if cols_to_drop:
        print(f"Dropping high missing columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Strategy 2: Fill remaining missing values
    # Numerical: mean imputation
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Categorical: mode imputation
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    print(f"✅ Missing values after treatment: {df_clean.isnull().sum().sum()}")
    return df_clean
```

---

### Phase 4: Outlier Treatment (IQR Method)

```python
def handle_outliers(df):
    df_clean = df.copy()
    
    # Apply IQR method to key numerical columns only
    numerical_cols = ['lead_time', 'adr', 'stays_in_weekend_nights', 
                     'stays_in_week_nights', 'adults', 'children', 'babies']
    
    # Filter to existing columns
    numerical_cols = [col for col in numerical_cols if col in df_clean.columns]
    
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers (don't remove - preserve data)
        outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outlier_count > 0:
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"Capped {outlier_count} outliers in {col}")
    
    return df_clean
```

---

### Phase 5: Final Data Preparation

**Remove Data Leakage & Low-Value Columns:**
```python
def prepare_for_modeling(df):
    df_final = df.copy()
    
    # Remove data leakage columns
    leakage_cols = ['reservation_status', 'reservation_status_date']
    existing_leakage = [col for col in leakage_cols if col in df_final.columns]
    if existing_leakage:
        df_final = df_final.drop(columns=existing_leakage)
        print(f"Removed leakage columns: {existing_leakage}")
    
    # Remove low-value columns (high cardinality, low predictive value)
    low_value_cols = ['company', 'agent']
    existing_low_value = [col for col in low_value_cols if col in df_final.columns]
    if existing_low_value:
        df_final = df_final.drop(columns=existing_low_value)
        print(f"Removed low-value columns: {existing_low_value}")
    
    # Encode categorical variables (simple label encoding)
    categorical_cols = df_final.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'is_canceled']
    
    for col in categorical_cols:
        if df_final[col].nunique() <= 20:  # Only encode reasonable categories
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
            print(f"Encoded: {col}")
    
    return df_final
```

---

## 🔧 EDA-Based Feature Engineering (High Priority Only)

**Add Essential Features from EDA Analysis:**
```python
def add_eda_features(df):
    df_features = df.copy()
    
    # 1. Lead time categorization (strongest predictor from EDA)
    if 'lead_time' in df_features.columns:
        df_features['lead_time_category'] = pd.cut(
            df_features['lead_time'], 
            bins=[0, 18, 69, 160, float('inf')], 
            labels=[0, 1, 2, 3]  # Immediate, Short, Medium, Long
        ).astype(int)
        
        # Lead time risk score
        df_features['lead_time_risk'] = pd.cut(
            df_features['lead_time'],
            bins=[0, 45, 113, float('inf')],
            labels=[0, 1, 2]  # Low, Medium, High risk
        ).astype(int)
    
    # 2. Total stay duration
    if all(col in df_features.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
        df_features['total_stay_duration'] = (
            df_features['stays_in_weekend_nights'] + df_features['stays_in_week_nights']
        )
    
    # 3. Family booking indicator
    if all(col in df_features.columns for col in ['children', 'babies']):
        df_features['is_family_booking'] = (
            (df_features['children'] > 0) | (df_features['babies'] > 0)
        ).astype(int)
    
    # 4. Total guests
    if all(col in df_features.columns for col in ['adults', 'children', 'babies']):
        df_features['total_guests'] = (
            df_features['adults'] + df_features['children'] + df_features['babies']
        )
    
    print("✅ Added 5 high-priority features based on EDA")
    return df_features
```

---

## 🚀 Complete Essential Pipeline

```python
def run_essential_preprocessing(input_path, output_path):
    """
    Complete essential preprocessing pipeline.
    """
    print("🚀 Starting Essential Preprocessing Pipeline...")
    start_time = datetime.now()
    
    # Phase 1: Load data
    df = load_data(input_path)
    original_shape = df.shape
    
    # Phase 2: Quality assessment
    missing_summary = assess_missing_values(df)
    validation_issues = validate_business_logic(df)
    
    # Phase 3: Handle missing values
    df = handle_missing_values(df)
    
    # Phase 4: Handle outliers
    df = handle_outliers(df)
    
    # Phase 5: Prepare for modeling
    df = prepare_for_modeling(df)
    
    # Add EDA-based features
    df = add_eda_features(df)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    
    # Summary
    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Preprocessing Complete!")
    print(f"📊 Shape: {original_shape} → {df.shape}")
    print(f"⏱️ Time: {processing_time:.2f} seconds")
    print(f"📁 Saved: {output_path}")
    print(f"📈 Final cancellation rate: {df['is_canceled'].mean():.3f}")
    
    return df

# Usage
if __name__ == "__main__":
    processed_df = run_essential_preprocessing(
        input_path='data/raw/hotel_booking.csv',
        output_path='data/processed/hotel_booking_preprocessed.csv'
    )
```

---

## 📋 Summary: Essential Requirements Only

**What's Included (Must-Have):**
1. ✅ Data loading and validation
2. ✅ Missing value assessment and treatment
3. ✅ Business logic validation (data leakage check)
4. ✅ Outlier detection and capping (IQR method)
5. ✅ Data leakage removal (reservation_status)
6. ✅ Categorical encoding (Label Encoder)
7. ✅ EDA-based feature engineering (5 high-priority features)

**What's Skipped (Optional):**
- ❌ Complex imputation methods (KNN, iterative)
- ❌ Class imbalance handling (SMOTE) - handle during modeling
- ❌ Advanced scaling/normalization - handle during modeling
- ❌ Extensive visualization - already done in EDA
- ❌ Advanced outlier methods (Isolation Forest, etc.)

**Expected Output:**
- Clean dataset ready for model training
- 5 new engineered features based on EDA findings
- No missing values
- No data leakage
- Outliers capped (not removed)
- Processing time: < 30 seconds

This simplified pipeline focuses on the essential preprocessing steps identified from your EDA analysis while maintaining academic rigor for NIB 7072 coursework.