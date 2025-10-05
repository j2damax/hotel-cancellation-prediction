import { API_BASE_URL } from '../constants/thresholds';
import type { 
  BookingInput, 
  PredictionResponse, 
  InterpretabilityResponse 
} from '../types/api';

const TIMEOUT_MS = 10000;

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

export async function predictBooking(
  booking: BookingInput
): Promise<PredictionResponse> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(booking),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  // The API returns a single prediction, but we wrap it in predictions array
  const result = await response.json();
  return {
    predictions: [result]
  };
}

let cachedInterpretability: InterpretabilityResponse | null = null;

export async function getInterpretability(): Promise<InterpretabilityResponse> {
  if (cachedInterpretability) {
    return cachedInterpretability;
  }

  const response = await fetchWithTimeout(`${API_BASE_URL}/model/interpretability`);

  if (!response.ok) {
    throw new Error(`Failed to fetch interpretability: ${response.status}`);
  }

  const data = await response.json();
  cachedInterpretability = data;
  return data;
}

export function clearInterpretabilityCache(): void {
  cachedInterpretability = null;
}
