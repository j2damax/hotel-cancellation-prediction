import { useState } from 'react';
import { getRiskColor } from '../lib/risk';
import type { PredictionHistoryEntry } from '../types/api';

interface ProbabilityHistoryChartProps {
  history: PredictionHistoryEntry[];
}

export function ProbabilityHistoryChart({ history }: ProbabilityHistoryChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (history.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Prediction History</h3>
        <p className="text-sm text-gray-500">
          No predictions yet. Submit a booking to see history.
        </p>
      </div>
    );
  }

  const maxProbability = Math.max(...history.map((h) => h.probability), 1);

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">
        Prediction History (Last {history.length})
      </h3>

      <div className="space-y-2">
        {history.map((entry, index) => (
          <div
            key={index}
            className="relative"
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-500 w-16">
                {entry.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
              <div className="flex-1 bg-gray-100 rounded-full h-6 relative overflow-hidden">
                <div
                  className={`h-6 rounded-full transition-all ${getRiskColor(
                    entry.risk
                  )}`}
                  style={{
                    width: `${(entry.probability / maxProbability) * 100}%`,
                  }}
                />
              </div>
              <div className="text-xs text-gray-700 font-medium w-12 text-right">
                {(entry.probability * 100).toFixed(1)}%
              </div>
            </div>

            {hoveredIndex === index && (
              <div className="absolute left-20 top-8 bg-gray-900 text-white text-xs rounded px-2 py-1 z-10 whitespace-nowrap">
                Probability: {(entry.probability * 100).toFixed(2)}% | Risk: {entry.risk}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
