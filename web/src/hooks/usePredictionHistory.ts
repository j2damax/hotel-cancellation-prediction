import { useState } from 'react';
import type { PredictionHistoryEntry, RiskLevel } from '../types/api';

const MAX_HISTORY = 25;

export function usePredictionHistory() {
  const [history, setHistory] = useState<PredictionHistoryEntry[]>([]);

  const add = (probability: number, risk: RiskLevel) => {
    setHistory((prev) => {
      const newEntry: PredictionHistoryEntry = {
        probability,
        risk,
        timestamp: new Date(),
      };
      const updated = [newEntry, ...prev];
      return updated.slice(0, MAX_HISTORY);
    });
  };

  return { history, add };
}
