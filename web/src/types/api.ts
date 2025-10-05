export interface BookingInput {
  lead_time: number;
  arrival_month: number;
  stays_weekend_nights: number;
  stays_week_nights: number;
  adults: number;
  children: number;
  is_repeated_guest: number;
  previous_cancellations: number;
  booking_changes: number;
  adr: number;
  required_car_parking_spaces: number;
  total_of_special_requests: number;
  deposit_type?: string;
  market_segment?: string;
}

export interface PredictionResult {
  prediction: number;
  probability: number;
  model_used: string;
}

export interface PredictionResponse {
  predictions: PredictionResult[];
}

export interface FeatureImportance {
  feature: string;
  human_readable: string;
  mean_abs_shap: number;
}

export interface FeatureContribution {
  feature: string;
  shap: number;
  human_readable: string;
}

export interface LocalExplanation {
  category: string;
  probability?: number;
  top_positive_contributors: FeatureContribution[];
  top_negative_contributors: FeatureContribution[];
}

export interface InterpretabilityResponse {
  champion_model?: string;
  shap_generated: boolean;
  shap_timestamp?: string;
  decision_threshold?: number;
  top_features: FeatureImportance[];
  local_examples: LocalExplanation[];
  feature_name_map: Record<string, string>;
  artifacts_available: string[];
}

export type RiskLevel = 'High' | 'Review' | 'Low';

export interface PredictionHistoryEntry {
  probability: number;
  risk: RiskLevel;
  timestamp: Date;
}
