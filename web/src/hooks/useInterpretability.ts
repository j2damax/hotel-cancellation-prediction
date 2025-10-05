import { useState, useEffect } from 'react';
import { getInterpretability } from '../lib/api';
import type { InterpretabilityResponse } from '../types/api';

export function useInterpretability() {
  const [data, setData] = useState<InterpretabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await getInterpretability();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load interpretability');
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { data, loading, error };
}
