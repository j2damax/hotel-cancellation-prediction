import { DECISION_THRESHOLD, HIGH_RISK_THRESHOLD } from '../constants/thresholds';
import type { RiskLevel } from '../types/api';

export function classifyRisk(probability: number): RiskLevel {
  if (probability >= HIGH_RISK_THRESHOLD) {
    return 'High';
  }
  if (probability >= DECISION_THRESHOLD) {
    return 'Review';
  }
  return 'Low';
}

export function getRiskColor(risk: RiskLevel): string {
  switch (risk) {
    case 'High':
      return 'bg-red-600';
    case 'Review':
      return 'bg-amber-500';
    case 'Low':
      return 'bg-emerald-600';
  }
}

export function getRiskTextColor(_risk: RiskLevel): string {
  return 'text-white';
}
