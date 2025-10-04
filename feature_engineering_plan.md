# Feature Engineering Plan - Hotel Cancellation Prediction
## Derived from EDA Analysis Results

Based on the comprehensive EDA analysis, here are the **new feature columns** we can introduce during preprocessing/feature engineering:

---

## 🎯 **1. Lead Time-Based Features** (High Impact - 0.293 correlation)

### **`lead_time_category`**
```python
# Categorize lead time into business-relevant buckets
def create_lead_time_category(lead_time):
    if lead_time <= 7:
        return 'last_minute'      # 0-7 days
    elif lead_time <= 30:
        return 'short_term'       # 8-30 days  
    elif lead_time <= 90:
        return 'medium_term'      # 31-90 days
    elif lead_time <= 180:
        return 'long_term'        # 91-180 days
    else:
        return 'very_long_term'   # 180+ days
```

### **`lead_time_risk_score`**
```python
# Risk score based on lead time distribution from EDA
# Canceled bookings average: 144.85 days vs Not Canceled: 79.98 days
def create_lead_time_risk_score(lead_time):
    if lead_time <= 45:       # Below median for not-canceled
        return 'low_risk'
    elif lead_time <= 113:    # Below median for canceled
        return 'medium_risk'
    else:                     # Above median for canceled
        return 'high_risk'
```

### **`is_early_booking`**
```python
# Binary indicator for very early bookings (potential higher cancellation)
df['is_early_booking'] = (df['lead_time'] > 180).astype(int)
```

---

## 🏨 **2. Stay Duration & Guest Composition Features**

### **`total_stay_duration`**
```python
# Total nights stayed (already valuable for revenue calculation)
df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
```

### **`weekend_ratio`**
```python
# Proportion of weekend nights in total stay
df['weekend_ratio'] = df['stays_in_weekend_nights'] / (df['total_stay_duration'] + 1e-8)
```

### **`total_guests`**
```python
# Total number of guests
df['total_guests'] = df['adults'] + df['children'] + df['babies']
```

### **`is_family_booking`**
```python
# Family indicator (children or babies present)
df['is_family_booking'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
```

### **`guest_type_category`**
```python
# Categorize based on guest composition
def create_guest_type(row):
    if row['babies'] > 0:
        return 'family_with_babies'
    elif row['children'] > 0:
        return 'family_with_children'
    elif row['adults'] == 1:
        return 'solo_traveler'
    elif row['adults'] == 2:
        return 'couple'
    else:
        return 'group'
```

---

## 💰 **3. Revenue & Pricing Features**

### **`revenue_per_night`**
```python
# Revenue per night calculation
df['revenue_per_night'] = df['adr'] * df['total_guests']
```

### **`total_booking_value`**
```python
# Total estimated revenue for the booking
df['total_booking_value'] = df['adr'] * df['total_stay_duration']
```

### **`price_per_person`**
```python
# ADR divided by total guests (price sensitivity indicator)
df['price_per_person'] = df['adr'] / (df['total_guests'] + 1e-8)
```

### **`is_high_value_booking`**
```python
# High value booking indicator (top quartile)
adr_75th = df['adr'].quantile(0.75)
df['is_high_value_booking'] = (df['adr'] >= adr_75th).astype(int)
```

---

## 📅 **4. Temporal & Seasonal Features**

### **`arrival_season`**
```python
# Season categorization from EDA seasonal analysis
season_mapping = {
    'December': 'winter', 'January': 'winter', 'February': 'winter',
    'March': 'spring', 'April': 'spring', 'May': 'spring',
    'June': 'summer', 'July': 'summer', 'August': 'summer',
    'September': 'autumn', 'October': 'autumn', 'November': 'autumn'
}
df['arrival_season'] = df['arrival_date_month'].map(season_mapping)
```

### **`is_peak_season`**
```python
# Based on EDA findings - high demand months
peak_months = ['July', 'August', 'May', 'June', 'September']  # From EDA analysis
df['is_peak_season'] = df['arrival_date_month'].isin(peak_months).astype(int)
```

### **`arrival_week_of_year`**
```python
# Week of year for more granular seasonal patterns
df['arrival_week_of_year'] = df['arrival_date_week_number']
```

### **`days_from_booking_to_arrival`** (alias for lead_time)
```python
# More intuitive name for lead_time
df['days_from_booking_to_arrival'] = df['lead_time']
```

---

## 🔄 **5. Customer Behavior Features**

### **`customer_loyalty_score`**
```python
# Composite score based on repeat guest status and previous bookings
def create_loyalty_score(row):
    score = 0
    if row['is_repeated_guest'] == 1:
        score += 3
    score += min(row['previous_bookings_not_canceled'], 5)  # Cap at 5
    score -= min(row['previous_cancellations'], 3)  # Penalty for cancellations
    return max(0, score)  # Ensure non-negative
```

### **`special_requests_per_guest`**
```python
# Special requests normalized by guest count
df['special_requests_per_guest'] = df['total_of_special_requests'] / (df['total_guests'] + 1e-8)
```

### **`booking_complexity_score`**
```python
# Complexity indicator (more complex = potentially more likely to cancel)
def create_complexity_score(row):
    score = 0
    score += row['total_of_special_requests'] * 0.5
    score += row['booking_changes'] * 1.0
    if row['required_car_parking_spaces'] > 0:
        score += 1
    if row['days_in_waiting_list'] > 0:
        score += 2
    return score
```

---

## 🏪 **6. Market Segment & Channel Features**

### **`is_online_booking`**
```python
# Online vs offline booking indicator
online_segments = ['Online TA', 'Direct']  # Based on market_segment
df['is_online_booking'] = df['market_segment'].isin(online_segments).astype(int)
```

### **`is_corporate_booking`**
```python
# Corporate/group booking indicator
corporate_segments = ['Corporate', 'Groups']
df['is_corporate_booking'] = df['market_segment'].isin(corporate_segments).astype(int)
```

### **`distribution_channel_risk`**
```python
# Risk score by distribution channel (can be derived from EDA analysis)
# This would need to be calculated from actual cancellation rates by channel
```

---

## 🎲 **7. Derived Risk Indicators**

### **`cancellation_risk_factors`**
```python
# Count of high-risk factors present
def count_risk_factors(row):
    risk_count = 0
    if row['lead_time'] > 113:  # Above median for canceled bookings
        risk_count += 1
    if row['total_of_special_requests'] == 0:  # Low engagement
        risk_count += 1
    if row['previous_cancellations'] > 0:  # History of cancellation
        risk_count += 1
    if row['booking_changes'] == 0:  # No modifications (less committed)
        risk_count += 1
    if row['is_repeated_guest'] == 0:  # New customer
        risk_count += 1
    return risk_count
```

### **`engagement_score`**
```python
# Customer engagement indicator (opposite of risk)
def create_engagement_score(row):
    score = 0
    score += row['total_of_special_requests'] * 2
    score += row['booking_changes']
    if row['is_repeated_guest'] == 1:
        score += 3
    if row['required_car_parking_spaces'] > 0:
        score += 1
    return score
```

---

## 📊 **8. Statistical Transformation Features**

### **`lead_time_log`**
```python
# Log transformation for lead_time (handle skewness identified in EDA)
df['lead_time_log'] = np.log1p(df['lead_time'])
```

### **`adr_zscore`**
```python
# Z-score normalization for ADR (handle outliers)
from scipy.stats import zscore
df['adr_zscore'] = zscore(df['adr'].fillna(df['adr'].median()))
```

---

## 🎯 **Implementation Priority (Based on EDA Correlations)**

### **High Priority** (Implement First):
1. `total_stay_duration` - fundamental business metric
2. `lead_time_category` - strongest predictor (0.293 correlation)
3. `is_family_booking` - guest composition insight
4. `total_booking_value` - revenue impact
5. `customer_loyalty_score` - repeat guest patterns

### **Medium Priority**:
6. `arrival_season` - seasonal patterns identified
7. `special_requests_per_guest` - engagement normalized
8. `cancellation_risk_factors` - composite risk score
9. `is_peak_season` - demand period indicator
10. `booking_complexity_score` - complexity patterns

### **Low Priority** (Experimental):
11. `lead_time_log` - statistical transformation
12. `weekend_ratio` - stay pattern refinement
13. `is_online_booking` - channel analysis
14. `engagement_score` - behavioral composite

---

## 🚀 **Next Steps**

1. **Start with High Priority features** during preprocessing
2. **Validate each feature** with correlation analysis post-creation
3. **A/B test features** in model training to measure impact
4. **Document feature definitions** for reproducibility
5. **Monitor feature importance** in trained models

This feature engineering plan is derived directly from your EDA findings and should significantly improve model performance by capturing business logic and patterns identified in the exploratory analysis.