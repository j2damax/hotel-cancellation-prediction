import { RiskBadge } from './RiskBadge';
import type { RiskLevel } from '../types/api';

interface ResultPanelProps {
  probability: number;
  risk: RiskLevel;
  modelVersion?: string;
}

export function ResultPanel({ probability, risk, modelVersion }: ResultPanelProps) {
  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">Prediction Result</h3>
      
      <div className="space-y-4">
        <div>
          <div className="text-sm text-gray-600 mb-1">Cancellation Probability</div>
          <div className="text-3xl font-bold text-gray-900">
            {(probability * 100).toFixed(2)}%
          </div>
        </div>

        <div>
          <div className="text-sm text-gray-600 mb-2">Risk Classification</div>
          <RiskBadge risk={risk} />
        </div>

        {modelVersion && (
          <div className="text-xs text-gray-500 mt-4">
            Model: {modelVersion}
          </div>
        )}
      </div>
    </div>
  );
}
