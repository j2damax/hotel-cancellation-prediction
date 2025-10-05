# Academic Report Draft

Date: 2025-10-05
Project: Hotel Booking Cancellation Prediction

## 3. Model Development

### 3.1 Model Families Implemented
- Linear: Logistic Regression (L2-regularized baseline) – interpretable, fast, establishes linear separability limit.
- Tree-Based: Random Forest – non-linear ensembles, robust to feature scaling, captures interactions.
- Boosting: XGBoost – gradient boosted trees with regularization & shrinkage; historically strongest performer on tabular, moderate cardinality categorical (after encoding) and class imbalance (weighting).
- Advanced: PyTorch MLP – feedforward neural network with dropout and batch normalization; serves as deep representation learner for potential non-linear manifolds.

### 3.2 Hyperparameter & Training Strategy
- Cross-Validation: Stratified 5-fold CV (primary metric: F1, tie-break: ROC-AUC) for robust selection.
- Class Imbalance: Addressed via class weights (no oversampling to preserve true distribution). SMOTE deliberately avoided to prevent synthetic bias in business-critical risk calibration.
- Encoding Strategy: Target encoding for categorical variables (selected for final run: `--categorical-strategy target`).
- Threshold Optimization: Post-hoc threshold sweep selects F1-optimal threshold (0.35) balancing precision and recall.

### 3.3 Rationale for Model Choices
| Model | Justification |
|-------|---------------|
| Logistic Regression | Baseline, interpretability, calibration-friendly, acts as performance floor |
| Random Forest | Handles heterogeneity & interactions; variance reduction via bagging |
| XGBoost | Strong bias-variance trade-off, handles sparse/encoded features efficiently, consistent top performer |
| PyTorch MLP | Tests potential gains from representation learning beyond tree ensembles |

### 3.4 Implementation Notes
- Unified preprocessing pipeline persisted (`preprocessor.pkl`).
- Dynamic champion selection: computed aggregate CV metrics; champion persisted as `champion_model.pkl`.
- MLflow used for run tracking (parameters, metrics, artifacts).

## 4. Evaluation & Comparison

### 4.1 Cross-Validation Aggregate Metrics (Final Full Run)
(From `artifacts/cv_metrics.json`)

| Model | Accuracy (µ±σ) | Precision (µ±σ) | Recall (µ±σ) | F1 (µ±σ) | ROC-AUC (µ±σ) |
|-------|----------------|-----------------|-------------|---------|--------------|
| LogisticRegression | 0.8107 ± 0.0026 | 0.8021 ± 0.0050 | 0.6490 ± 0.0051 | 0.7174 ± 0.0041 | 0.8896 ± 0.0018 |
| RandomForest | 0.8496 ± 0.0025 | 0.8743 ± 0.0067 | 0.6937 ± 0.0093 | 0.7735 ± 0.0048 | 0.9277 ± 0.0010 |
| XGBoost | 0.8612 ± 0.0016 | 0.8386 ± 0.0024 | 0.7743 ± 0.0051 | 0.8052 ± 0.0028 | 0.9376 ± 0.0010 |

Champion: XGBoost (highest mean F1; ROC-AUC tie-break not needed).

### 4.2 Hold-Out Performance (Champion)
(From `champion_meta.json` holdout metrics)
- Accuracy: 0.8614
- Precision: 0.8417
- Recall: 0.7707
- F1: 0.8047
- ROC-AUC: 0.9384

### 4.3 Threshold Optimization
Optimal F1 threshold: 0.35 (F1=0.8114; Precision=0.7664; Recall=0.8620) – chosen to slightly favor recall (capturing risky cancellations) while maintaining acceptable precision to limit false alarms.

Trade-off: At default 0.50 threshold, precision increases (0.8417) but recall drops (0.7707), reducing potential proactive interventions.

### 4.4 Classification Report (Per Class)
(From `classification_report.json`)
- Class 0 (Not Canceled) F1: 0.8926 (support=15033)
- Class 1 (Canceled) F1: 0.8047 (support=8845)
- Macro F1: 0.8486; Weighted F1: 0.8600

Misclassification emphasis: Slightly lower recall on positive (canceled) class motivates threshold tuning.

### 4.5 Error Analysis Summary
- False Negatives: Concentrated among mid lead_time & moderate special requests – potential value in segment-specific thresholding.
- False Positives: Bookings with long lead_time and high price variance; may benefit from temporal decay feature or channel segmentation refinement.
- Confusion Matrix captured in `confusion_matrix.png` (see appendix).

### 4.6 Adjustments Applied
- Class weights for imbalance (approx 32–33% cancellations) maintained natural distribution.
- Threshold tuning applied for business-driven operating point.
- Categorical target encoding improved tree/boosting stability compared to one-hot (empirically observed lower variance in CV metrics).

### 4.7 Experiment Tracking (MLflow)
Tracked:
- Parameters: encoding strategy, folds, model hyperparameters (default or tuned base settings).
- Metrics: fold-level + aggregate (F1 primary), holdout metrics, threshold sweep results (logged via artifacts).
- Artifacts: model pkl, preprocessing pipeline, ROC/PR curves JSON, confusion matrix, SHAP visuals, feature importance JSON, local explanations.

## 5. Interpretability & Insights

### 5.1 Explainability Techniques
- SHAP (TreeExplainer for XGBoost) for global + local attribution.
- Local examples: true positive / false positive / false negative cases stored in `shap_values_sample.json`.
- Feature name mapping for human-readable reporting (`feature_name_map.json`).

### 5.2 Key Influential Features (Indicative)
(Refer to `feature_importance.json` & SHAP plots.) Common top drivers observed:
- lead_time
- total_of_special_requests
- market_segment / distribution_channel encodings
- adr (average daily rate)
- booking_changes
- is_repeated_guest

### 5.3 Business Interpretation
- High lead_time + specific market segments elevate risk: supports overbooking calibration for early reservations.
- Low special requests + first-time guests correlate with higher cancellation probability – opportunity for targeted pre-arrival engagement.
- Moderate booking_changes increase risk (instability as behavioral signal). Suggest flagging bookings with multiple modifications.

### 5.4 Potential Policy Actions
| Insight | Action |
|---------|--------|
| Long lead time, high ADR, new guest | Proactive confirmation email / flexible incentive |
| Low special requests pattern | Encourage add-ons (stickiness) |
| Multiple booking changes | Escalate to revenue manager for review |
| Channel-specific high risk | Adjust channel overbooking factor |

### 5.5 Additional Explainability Options (Future)
- Partial Dependence / ICE plots for top 5 features
- Global surrogate model for narrative explanation
- SHAP-based drift monitoring over time

## 6. Critical Reflection

### 6.1 Dataset Limitations
- Potential leakage if reservation status fields processed incorrectly (mitigated by dropping post-outcome fields).
- Country & channel encodings may proxy socio-economic or geographic bias – requires fairness audit.
- Temporal drift risk: booking behavior changes (seasonality, macro events) not fully modeled.

### 6.2 Ethical & Bias Considerations
- Risk of over-personalization affecting pricing fairness.
- Country-based signals should be aggregated or anonymized to avoid discriminatory policy.
- Transparency: Provide SHAP-derived reasons in customer-facing interventions sparingly to avoid strategic manipulation.

### 6.3 Generalizability
- Trained on a specific distribution (hotels dataset); transferring to different regions or property classes requires re-training with adaptation features (e.g., property_type, distribution shift indicators).

### 6.4 Future Extensions
- Probability calibration (isotonic/Platt) for improved economic decision thresholds.
- Time-aware modeling (lead time decay, booking window buckets).
- Ensemble stacking (blend XGBoost + calibrated logistic meta-model).
- Incorporate cancellation cost estimation to optimize expected value, not just F1.

## 7. Deployment

### 7.1 Deployment Architecture
- FastAPI inference service loading `champion_model.pkl` + `preprocessor.pkl`.
- Containerization via Docker (`Dockerfile`, `docker-compose.yml` with MLflow UI sidecar).
- MLflow for experiment artifacts (potential future Model Registry integration).
- CI: GitHub Actions runs tests on push (skips live API tests by default).

### 7.2 Versioning & CI/CD
- Semantic commit messages; artifacts tied to MLflow run IDs.
- Potential enhancement: tag champion model run and push image with matching tag.

### 7.3 Monitoring & Ops Considerations
| Aspect | Approach |
|--------|----------|
| Performance Drift | Periodic SHAP distribution comparison vs baseline |
| Data Quality | Validate schema using feature contract before scoring |
| Logging | Store prediction + confidence + threshold decision outcome |
| Threshold Review | Recompute threshold_sweep quarterly |

### 7.4 Scaling Options
- Horizontal scaling with container orchestration (ECS/EKS/Kubernetes).
- Add Redis cache for preprocessing invariants if latency spikes.
- Optional GPU not required (tree model champion). MLP remains CPU-sufficient.

### 7.5 Security & Compliance
- Future: add API key / JWT middleware.
- Environment variable secrets (.env) – replace with AWS Secrets Manager in production.

## Appendix (Artifacts)
- confusion_matrix.png
- roc_curve.json / pr_curve.json
- shap_summary.png / shap_importance_bar.png
- feature_importance.json / shap_values_sample.json
- champion_meta.json / cv_metrics.json / threshold_sweep.csv

---
Draft complete. Replace indicative feature rankings with final SHAP ordering before publication.
