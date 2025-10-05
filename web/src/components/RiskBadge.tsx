import { getRiskColor, getRiskTextColor } from '../lib/risk';
import type { RiskLevel } from '../types/api';

interface RiskBadgeProps {
  risk: RiskLevel;
}

export function RiskBadge({ risk }: RiskBadgeProps) {
  return (
    <span
      className={`inline-block px-2 py-1 text-xs font-semibold rounded-full ${getRiskColor(
        risk
      )} ${getRiskTextColor(risk)}`}
    >
      {risk}
    </span>
  );
}
