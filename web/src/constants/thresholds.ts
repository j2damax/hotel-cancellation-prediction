export const DECISION_THRESHOLD = parseFloat(
  import.meta.env.VITE_DECISION_THRESHOLD || '0.35'
);

export const HIGH_RISK_THRESHOLD = parseFloat(
  import.meta.env.VITE_HIGH_RISK_THRESHOLD || '0.55'
);

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
