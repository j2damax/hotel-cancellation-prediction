import { useState } from 'react';
import { PredictionForm } from './components/PredictionForm';
import { ResultPanel } from './components/ResultPanel';
import { FeatureContributions } from './components/FeatureContributions';
import { ProbabilityHistoryChart } from './components/ProbabilityHistoryChart';
import { usePredictionHistory } from './hooks/usePredictionHistory';
import { useInterpretability } from './hooks/useInterpretability';
import { predictBooking } from './lib/api';
import { classifyRisk } from './lib/risk';
import type { BookingInput, PredictionResult } from './types/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const { history, add } = usePredictionHistory();
  const { data: interpretability, loading: interpLoading, error: interpError } = useInterpretability();

  const handleSubmit = async (booking: BookingInput) => {
    setLoading(true);
    setError(null);

    try {
      const response = await predictBooking(booking);
      const prediction = response.predictions[0];
      
      setResult(prediction);
      
      const risk = classifyRisk(prediction.probability);
      add(prediction.probability, risk);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const modelVersion = result?.model_used || interpretability?.champion_model || 'unknown';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Hotel Cancellation Prediction
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Model: {modelVersion}
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Form */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h2 className="text-xl font-semibold mb-4">Booking Details</h2>
              <PredictionForm onSubmit={handleSubmit} loading={loading} />
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800">
                  ⚠️ {error}
                </p>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="space-y-6">
            {result && (
              <ResultPanel
                probability={result.probability}
                risk={classifyRisk(result.probability)}
                modelVersion={result.model_used}
              />
            )}

            {interpLoading ? (
              <div className="bg-white p-6 rounded-lg border border-gray-200">
                <p className="text-sm text-gray-500">Loading feature importance...</p>
              </div>
            ) : interpError ? (
              <div className="bg-white p-6 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-2">Feature Contributions</h3>
                <p className="text-sm text-gray-500">
                  Failed to load interpretability data
                </p>
              </div>
            ) : (
              <FeatureContributions 
                features={interpretability?.top_features || []} 
              />
            )}

            <ProbabilityHistoryChart history={history} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center">
          <p className="text-sm text-gray-600">
            <a
              href="https://github.com/j2damax/hotel-cancellation-prediction"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              GitHub Repository
            </a>
            {' | '}
            Version: {modelVersion}
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
